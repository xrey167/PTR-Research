"""Run a bounded Pod socket server for cross-host integration tests."""
from __future__ import annotations
import argparse, json, signal, time
from neural_pods.pod_protocol import PodTransport
from neural_pods.pod_socket import PodSocketServer


def main(host: str, port: int, seconds: float):
    transport = PodTransport(secret=b"cross-host-benchmark")
    transport.register("research", "lookup", lambda payload, request: {"server": "xrserver", "found": True}, manifest_hash="research-v1")
    server = PodSocketServer(transport, host=host, port=port); address = server.start()
    print(json.dumps({"host": address[0], "port": address[1]}), flush=True)
    try: time.sleep(seconds)
    finally: server.close()


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--host",default="0.0.0.0"); p.add_argument("--port",type=int,default=39123); p.add_argument("--seconds",type=float,default=120); a=p.parse_args(); main(a.host,a.port,a.seconds)
