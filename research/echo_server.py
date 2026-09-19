import socket
import threading

server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", 45779))
server.listen(16)
print("up", flush=True)


def serve():
    while True:
        conn, _ = server.accept()
        def handle(c):
            try:
                while True:
                    data = c.recv(4096)
                    if not data:
                        return
                    c.sendall(data)
            except OSError:
                pass
            finally:
                c.close()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


serve()
