use pyo3::prelude::*;
use raft::prelude::*;
use raft::storage::MemStorage;
use slog::Drain;
use protobuf::Message as ProtoMessage;
use bytes::Bytes;

/// Thin PyO3 binding around raft-rs' RawNode. Durable storage and transport
/// stay outside this object and are supplied by the Python Pod control plane.
#[pyclass]
struct RaftNode {
    node: RawNode<MemStorage>,
    pending_ready: Option<Ready>,
}

fn logger() -> slog::Logger {
    let drain = slog::Discard.fuse();
    slog::Logger::root(drain, slog::o!())
}

#[pymethods]
impl RaftNode {
    #[new]
    #[pyo3(signature = (id, peers=None, entries=None, hard_state=None))]
    fn new(id: u64, peers: Option<Vec<u64>>, entries: Option<Vec<(u64, u64, Vec<u8>)>>, hard_state: Option<(u64, u64, u64)>) -> PyResult<Self> {
        if id == 0 { return Err(pyo3::exceptions::PyValueError::new_err("id must be positive")); }
        let mut voters = peers.unwrap_or_else(|| vec![id]);
        if !voters.contains(&id) { voters.push(id); }
        voters.sort_unstable(); voters.dedup();
        let storage = MemStorage::new_with_conf_state(ConfState::from((voters, vec![])));
        // Initial state must be in the storage BEFORE RawNode exists; filling
        // the unstable region afterwards panics in raft-rs' log_unstable.
        if let Some(entries) = entries {
            if !entries.is_empty() {
                let es: Vec<Entry> = entries.into_iter().map(|(index, term, data)| {
                    let mut e = Entry::default();
                    e.set_index(index);
                    e.set_term(term);
                    e.set_data(Bytes::from(data));
                    e
                }).collect();
                storage.wl().append(&es)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
        }
        if let Some((term, vote, commit)) = hard_state {
            let mut hs = HardState::default();
            hs.set_term(term);
            hs.set_vote(vote);
            hs.set_commit(commit);
            storage.wl().set_hardstate(hs);
        }
        let cfg = Config {
            id,
            election_tick: 10,
            heartbeat_tick: 3,
            max_size_per_msg: 1024 * 1024,
            max_inflight_msgs: 256,
            ..Default::default()
        };
        RawNode::new(&cfg, storage, &logger())
            .map(|node| Self { node, pending_ready: None })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn campaign(&mut self) -> PyResult<Vec<Vec<u8>>> {
        self.node.campaign()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        self.drain_ready()
    }

    fn propose(&mut self, data: &[u8]) -> PyResult<Vec<Vec<u8>>> {
        self.node.propose(vec![], data.to_vec())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        self.drain_ready()
    }

    fn tick(&mut self) -> PyResult<Vec<Vec<u8>>> {
        self.node.tick();
        self.drain_ready()
    }

    /// Tick and poll a Ready without acknowledging it. The caller must send
    /// the pending messages (election votes, heartbeats) and call
    /// `ack_ready_messages` afterwards — `tick()` would silently drop them.
    fn tick_pending(&mut self) -> PyResult<Option<(Vec<(u64, u64, Vec<u8>)>, Option<(u64, u64, u64)>, Vec<(u64, u64, Vec<u8>)>)>> {
        self.node.tick();
        self.poll_ready()
    }

    /// Propose without applying the Ready. The caller must persist the
    /// returned entries/HardState and call `ack_ready` afterwards.
    fn propose_pending(&mut self, data: &[u8]) -> PyResult<Option<(Vec<(u64, u64, Vec<u8>)>, Option<(u64, u64, u64)>, Vec<(u64, u64, Vec<u8>)>)>> {
        self.node.propose(vec![], data.to_vec())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        self.poll_ready()
    }

    fn campaign_pending(&mut self) -> PyResult<Option<(Vec<(u64, u64, Vec<u8>)>, Option<(u64, u64, u64)>, Vec<(u64, u64, Vec<u8>)>)>> {
        self.node.campaign()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        self.poll_ready()
    }

    fn poll_ready(&mut self) -> PyResult<Option<(Vec<(u64, u64, Vec<u8>)>, Option<(u64, u64, u64)>, Vec<(u64, u64, Vec<u8>)>)>> {
        if self.pending_ready.is_some() || !self.node.has_ready() { return Ok(None); }
        let ready = self.node.ready();
        let entries = ready.entries().iter().map(|e| (e.index, e.term, e.data.to_vec())).collect();
        let hs = ready.hs().map(|h| (h.term, h.vote, h.commit));
        let committed = ready.committed_entries().iter().map(|e| (e.index, e.term, e.data.to_vec())).collect();
        self.pending_ready = Some(ready);
        Ok(Some((entries, hs, committed)))
    }

    /// Acknowledge a Ready after the external WAL has durably persisted it.
    fn ack_ready(&mut self) -> PyResult<Vec<Vec<u8>>> {
        Ok(self.ack_ready_messages()?.0)
    }

    /// Persist and advance a Ready, returning committed payloads plus the
    /// messages released by `advance` (needed by candidate/leader paths).
    fn ack_ready_messages(&mut self) -> PyResult<(Vec<Vec<u8>>, Vec<(u64, Vec<u8>)>)> {
        let mut ready = self.pending_ready.take().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no pending Ready"))?;
        if !ready.entries().is_empty() {
            self.node.mut_store().wl().append(ready.entries())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }
        if let Some(hs) = ready.hs() { self.node.mut_store().wl().set_hardstate(hs.clone()); }
        let committed = ready.take_committed_entries();
        let light = self.node.advance(ready);
        let messages = light.messages().iter().map(|m| {
            m.write_to_bytes().map(|bytes| (m.to, bytes))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }).collect::<PyResult<Vec<_>>>()?;
        let committed = committed.into_iter().filter(|e| e.get_entry_type() == EntryType::EntryNormal && !e.data.is_empty())
            .map(|e| e.data.to_vec()).collect();
        Ok((committed, messages))
    }

    /// Restore a durable log and hard state after a crash-restart. Must be
    /// called before the node rejoins the cluster; without it, a restarted
    /// node has an empty log and can only be caught up via snapshot.
    #[pyo3(signature = (entries, hard_state=None))]
    fn restore(&mut self, entries: Vec<(u64, u64, Vec<u8>)>, hard_state: Option<(u64, u64, u64)>) -> PyResult<()> {
        let es: Vec<Entry> = entries.into_iter().map(|(index, term, data)| {
            let mut e = Entry::default();
            e.set_index(index);
            e.set_term(term);
            e.set_data(Bytes::from(data));
            e
        }).collect();
        if !es.is_empty() {
            self.node.mut_store().wl().append(&es)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }
        if let Some((term, vote, commit)) = hard_state {
            let mut hs = HardState::default();
            hs.set_term(term);
            hs.set_vote(vote);
            hs.set_commit(commit);
            self.node.mut_store().wl().set_hardstate(hs);
        }
        Ok(())
    }

    /// Return outbound Raft messages from the pending Ready as protobuf bytes.
    /// The network adapter sends these bytes to the placement-selected peer
    /// before acknowledging the Ready.
    fn pending_messages(&self) -> PyResult<Vec<Vec<u8>>> {
        let ready = self.pending_ready.as_ref().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no pending Ready"))?;
        ready.messages().iter().chain(ready.persisted_messages().iter()).map(|m| m.write_to_bytes()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))).collect()
    }

    fn pending_message_targets(&self) -> PyResult<Vec<u64>> {
        let ready = self.pending_ready.as_ref().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no pending Ready"))?;
        Ok(ready.messages().iter().chain(ready.persisted_messages().iter()).map(|m| m.to).collect())
    }

    /// Deliver one protobuf-encoded Raft message from a peer.
    fn step_message(&mut self, data: &[u8]) -> PyResult<Option<(Vec<(u64, u64, Vec<u8>)>, Option<(u64, u64, u64)>, Vec<(u64, u64, Vec<u8>)>)>> {
        let message = Message::parse_from_bytes(data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.node.step(message)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        self.poll_ready()
    }

    fn status(&self) -> (u64, u64, String) {
        let status = self.node.status();
        (status.id, status.hs.term, format!("{:?}", status.ss))
    }
}

impl RaftNode {
    fn drain_ready(&mut self) -> PyResult<Vec<Vec<u8>>> {
        if self.pending_ready.is_none() { self.poll_ready()?; }
        if self.pending_ready.is_some() { self.ack_ready() } else { Ok(Vec::new()) }
    }
}

#[pymodule]
fn neural_pods_raft(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RaftNode>()?;
    Ok(())
}
