"""Transport comparison: length-framed TCP vs gRPC unary, same host.

Both transports carry the same 256-byte payload; latency and throughput over
ROUNDS sequential round trips. Informs the deferred grpc-rs decision
(RUST-TRANSPORT-CONCURRENCY doc) with single-host numbers.
"""
import json
import socket
import struct
import subprocess
import sys
import threading
import time
from concurrent import futures
from pathlib import Path

import grpc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import echo_pb2
import echo_pb2_grpc

PAYLOAD = b"x" * 256
ROUNDS = 2000
TCP_PORT = 45777
GRPC_PORT = 45778


def tcp_server(port, ready):
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(8)
    ready.set()
    conn, _ = server.accept()
    with conn:
        while True:
            header = conn.recv(4)
            if not header:
                return
            size = struct.unpack("!I", header)[0]
            data = b""
            while len(data) < size:
                data += conn.recv(size - len(data))
            conn.sendall(struct.pack("!I", len(data)) + data)


def tcp_bench():
    ready = threading.Event()
    threading.Thread(target=tcp_server, args=(TCP_PORT, ready), daemon=True).start()
    ready.wait()
    out = socket.create_connection(("127.0.0.1", TCP_PORT))
    lat = []
    for _ in range(ROUNDS):
        start = time.perf_counter()
        out.sendall(struct.pack("!I", len(PAYLOAD)) + PAYLOAD)
        header = out.recv(4)
        size = struct.unpack("!I", header)[0]
        data = b""
        while len(data) < size:
            data += out.recv(size - len(data))
        lat.append((time.perf_counter() - start) * 1000)
    out.close()
    lat.sort()
    return {"p50_ms": lat[ROUNDS // 2], "p99_ms": lat[int(ROUNDS * 0.99)],
            "req_per_s": round(ROUNDS / (sum(lat) / 1000), 1)}


class EchoServicer(echo_pb2_grpc.EchoServicer):
    def RoundTrip(self, request, context):
        return echo_pb2.Frame(payload=request.payload)


def grpc_bench():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    echo_pb2_grpc.add_EchoServicer_to_server(EchoServicer(), server)
    server.add_insecure_port(f"127.0.0.1:{GRPC_PORT}")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{GRPC_PORT}")
    stub = echo_pb2_grpc.EchoStub(channel)
    stub.RoundTrip(echo_pb2.Frame(payload=PAYLOAD))  # warm
    lat = []
    for _ in range(ROUNDS):
        start = time.perf_counter()
        stub.RoundTrip(echo_pb2.Frame(payload=PAYLOAD))
        lat.append((time.perf_counter() - start) * 1000)
    lat.sort()
    server.stop(0)
    return {"p50_ms": lat[ROUNDS // 2], "p99_ms": lat[int(ROUNDS * 0.99)],
            "req_per_s": round(ROUNDS / (sum(lat) / 1000), 1)}


if __name__ == "__main__":
    print(json.dumps({"tcp": tcp_bench(), "grpc_unary": grpc_bench()}, indent=2))
