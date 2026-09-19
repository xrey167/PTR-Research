import sys, time
sys.path.insert(0, "/home/xrey/neural-pods")
from neural_pods.mesh import MeshEndpoint
responder = MeshEndpoint('10.50.0.121', 'qos0-responder')
responder.subscribe('np/qos0-responder/ping', lambda t,e: responder._client.publish('np/qos0-responder/pong', str(e.get('body',{}).get('ts','x')).encode(), qos=0))
time.sleep(0.5)
pings = MeshEndpoint('10.50.0.121', 'qos0-pinger')
lat = []
pongs = []
pings.subscribe('np/qos0-responder/pong', lambda t,e: pongs.append(time.perf_counter()))
time.sleep(0.5)
for i in range(50):
    start = time.perf_counter()
    pings._client.publish('np/qos0-responder/ping', str(i), qos=0)
    deadline = start + 2
    while len(pongs) <= i and time.perf_counter() < deadline:
        time.sleep(0.0002)
    if len(pongs) > i:
        lat.append((pongs[i] - start)*1000)
lat.sort()
print('qos0: ok', len(lat), '/50, p50', round(lat[len(lat)//2],2) if lat else None, 'ms')
responder.close(); pings.close()
