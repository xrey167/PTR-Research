import json,time
from neural_pods.pod_protocol import PodRequest,PodTransport,PodFanout
from neural_pods.resource_runtime import RequestTracker

def run(n=10000):
 t= RequestTracker(max_records=n)
 tr=PodTransport(tracker=t)
 tr.register('model','infer',lambda payload,req:{'value':payload['value'],'input_tokens':4,'output_tokens':8,'cache_hit':payload['value']%2==0})
 req=[PodRequest('router','model','infer',{'value':i}) for i in range(n)]
 s=time.perf_counter(); rs=PodFanout(tr,max_workers=32).dispatch(req); e=time.perf_counter()
 print(json.dumps({'requests':n,'qps':n/(e-s),'correct':sum(r.ok and r.payload.get('value')==i for i,r in enumerate(rs)),'errors':sum(not r.ok for r in rs),'tracker':t.stats()},indent=2))
run()
