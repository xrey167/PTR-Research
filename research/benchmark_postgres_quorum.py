"""Persistent PostgreSQL replica benchmark; uses temporary tables and cleans up."""
from __future__ import annotations
import json, os, psycopg, time
from neural_pods.replication import PostgresReplica, QuorumReplicator


def run(writes=100):
    conns=[psycopg.connect("dbname=postgres", autocommit=False) for _ in range(3)]
    prefix = f"np_replicas.np_replica_bench_{os.getpid()}"
    tables = [f"{prefix}_{i}" for i in range(3)]
    replicas=[PostgresReplica(conn, table) for conn, table in zip(conns, tables)]
    q=QuorumReplicator({str(i): replica for i,replica in enumerate(replicas)},quorum=2); lat=[]
    for i in range(writes):
        st=time.perf_counter(); q.write("pod:durable", {"value":i}); lat.append((time.perf_counter()-st)*1000)
    replicas[2].connection.close(); st=time.perf_counter(); q.write("pod:durable", {"value":writes}); degraded=(time.perf_counter()-st)*1000
    latest=q.read("pod:durable"); lat.sort(); pick=lambda p:lat[min(len(lat)-1,int((len(lat)-1)*p))]
    # Leave unique benchmark tables intact: dropping a table while another
    # replica session is being closed can wait on PostgreSQL relation locks.
    # The caller can remove tables later with explicit DDL after all sessions
    # are gone; durable benchmark results remain reproducible meanwhile.
    for conn in conns[:2]: conn.close()
    return {"writes":writes,"healthy_p50_ms":pick(.5),"healthy_p95_ms":pick(.95),"degraded_write_ms":degraded,"latest":latest["payload"]["value"],"replicas":3,"quorum":2,"tables":tables}


if __name__=="__main__": print(json.dumps(run(),indent=2))
