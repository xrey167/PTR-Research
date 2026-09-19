from neural_pods.routing_harness import ModelCandidate, RoutingHarness, TaskDemand

def test_harness_routes_to_capability_fit_and_records_verified_feedback():
    h=RoutingHarness([
      ModelCandidate('qwen-reader','qwen3.5',('retrieval','multihop'),latency_ms=100),
      ModelCandidate('neohorse','qwen3.5',('agentic','coding','multihop'),latency_ms=120),
    ])
    d=TaskDemand('t1',('agentic','coding'),max_latency_ms=200)
    chosen=h.select(d); assert chosen.model_id=='neohorse'
    h.record(d,chosen,tool_calls=('search','test'),outcome='success',verified=True)
    h.record(TaskDemand('t2',('coding',)),chosen,tool_calls=(),outcome='failed',verified=False)
    assert h.capability_feedback()['coding']=={'verified':1.0,'success':1.0,'success_rate':1.0}

def test_harness_respects_latency_budget():
    h=RoutingHarness([ModelCandidate('slow','x',('math',),latency_ms=1000),ModelCandidate('fast','y',('math',),latency_ms=10)])
    assert h.select(TaskDemand('t',('math',),max_latency_ms=50)).model_id=='fast'
