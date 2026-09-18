# Multi-Host-Cluster auf LXD — 2026-09-17

## Topologie

3 LXD-Container (Ubuntu 24.04) auf xrserver, bridged Netz `lxdbr0` (10.50.0.0/24):

| Node | IP | Rolle im Benchmark |
|---|---|---|
| np-node1 | 10.50.0.121 | Initiatorkandidat, später Failover-Opfer |
| np-node2 | 10.50.0.153 | Follower → neuer Leader nach Failover |
| np-node3 | 10.50.0.7 | Follower → Leader nach zweitem Failover |

Container mounten den Projektcode (`/home/xrey/neural-pods`), das
LoRA-venv und den uv-Python-Interpreter read-only vom Host. Postgres 16 ist
per Cloud-Init installiert (für Paket A3). Storage-Pool `np-pool-ai` liegt
auf `/srv/ai/workspaces/lxd-pool` (Root-Laufwerk war 100 % voll).

Reproduktion: `research/lxd_cluster.sh` (idempotent).

## Binding-Erweiterungen (neural_pods_raft 0.1.0, neu gebaut)

- `tick_pending()` — tick + poll ohne Auto-Ack. Das alte `tick()` verschluckte
  die Ready-Nachrichten (RequestVote/Heartbeat) über `drain_ready()`, sodass
  Auto-Election und Heartbeats nie das Netz erreichten.
- `restore(entries, hard_state)` — Crash-Recovery: WAL in MemStorage laden.

Build: `CARGO_TARGET_DIR=/srv/ai/workspaces/raft-target PYO3_PYTHON=<venv python> \
python -m maturin build --release -i <venv python> --target-dir /srv/ai/workspaces/raft-target`
(wichtig: ohne `PYO3_PYTHON` pickt pyo3 das System-Python 3.14 und bricht ab)

## Node-Server (`research/raft_node_server.py`)

- Raft-Port 45300 (4-Byte-Längenframing, protobuf), Control-Port 45400 (JSON)
- WAL `/tmp/raft-node-<id>.wal`: fsync vor jedem Ack (persist-before-ack)
- Peer-Verbindungen: TCP-Keepalive + 30 s TTL-Recycling (Halb-offene
  Verbindungen nach Container-Restart werden sonst nie bemerkt)

## Gemessene Ergebnisse (`research/runs/raft-multihost-20260917.json`)

| Phase | Ergebnis |
|---|---|
| Leader-Wahl über Netz | < 0,1 s |
| 100 Proposals, Replikation auf alle 3 Knoten | 100/100, 3.338 prop/s |
| Leader-Container hart gestoppt (lxc stop) | neuer Leader in < 1 s |
| Quorum-Writes nach Failover (2 von 3 Knoten) | 50/50 |
| Rejoin des Neustart-Knotens | **grün** (State-Transfer, 2026-09-19) |

## Rejoin-Catch-up (gelöst, 2026-09-19)

Der Leader liefert keinen Backfill an einen Knoten mit kurzem Log: raft-rs
verwirft den Candidate-Log bei der Wahl, der Neustart-Knoten gerät in eine
Vote-Retry-Schleife (AppendEntries-Nachschub kommt nie). Lösung: out-of-band
State-Transfer vor dem Wiedereintritt —

1. `export_state`-Control-Befehl am Cluster-Mitglied: kompletter Log +
   HardState aus dessen WAL.
2. Der Neustart-Knoten startet mit `--bootstrap-from <Mitglied>`, zieht den
   Stand und gibt ihn in den **RaftNode-Konstruktor** (`entries`,
   `hard_state`-Parameter, neu im Binding). Ein nachträgliches `restore()`
   nach der RawNode-Konstruktion panikt in raft-rs' `log_unstable` — der
   State muss vor dem Node-Start in den Storage.
3. Der Wiedereintritt ist dann ein normaler Follower-Weiterschrieb; Node
   reapplied 150 Einträge und bleibt im Quorum.

Vollständiger Benchmark-Lauf 2026-09-19: Wahl 0,001 s · 100/100 Replikation ·
Failover mit 50/50 Quorum-Writes · node_rejoined true. Architecture-Gate
21/21 Checks grün.

## Betrieb

```bash
research/lxd_cluster.sh                          # Cluster hochfahren (idempotent)
cd /home/xrey/neural-pods
PYTHONPATH=. /srv/ai/workspaces/llm-lora/.venv/bin/python \
  research/benchmark_raft_cluster_multihost.py   # Benchmark inkl. Failover
```
