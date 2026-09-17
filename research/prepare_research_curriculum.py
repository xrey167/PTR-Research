"""Generate the research curriculum derived from the project's architecture.

Each capability has 50 examples in every split. Targets are canonical, typed
actions/decisions so the set can train the Qwen reader and router without
pretending synthetic facts are real company data.
"""
import hashlib, json, math, shutil, sys
from pathlib import Path

src, out = map(Path, sys.argv[1:3])
shutil.copytree(src, out)

CAPS = {
  "pod_lifecycle": ("Resolve the active generation for research Pod {i} after a superseding update.", lambda i: {"action":"resolve_generation","knowledge_key":f"knowledge:research:{i}","status":"active","generation":f"g{(i%9)+1}"}),
  "tool_selection": ("Which research tool should handle this request: {topic}?", lambda i: {"tool":["ann_search","bm25_search","metadata_filter","graph_neighbors","registry_snapshot","calculator","workspace_search"][i%7],"contract":"research_tool_v2"}),
  "contextual_chunking": ("Create a retrieval chunk for section {i} while retaining its heading, table labels, and leading statement.", lambda i: {"action":"contextual_chunk","preserve":["heading_path","table_headers","leading_statement"],"leaf_only":True,"chunk_id":f"chunk:{i}"}),
  "statement_chaining": ("Resolve the antecedent chain for statement {i} before embedding it.", lambda i: {"action":"statement_chain","max_hops":(i%4)+1,"include_dependencies":True,"leaf_only":True}),
  "namespace_cache": ("Choose the storage and cache policy for namespace {i} during a bursty training run.", lambda i: {"namespace":f"pod-ns-{i}","cache":"memory_then_nvme_then_object","consistency":"strong","branch_before_update":True}),
  "routing_ranker": ("Route query {i} to a Pod using entity, type, tag, generation, and learned representation.", lambda i: {"action":"route","signals":["entity","pod_type","tags","generation","pod_embedding"],"hard_filter":"active_generation","ranker":"hybrid_ann_bm25"}),
  "index_performance": ("Build an index for workload {i} with many updates and high-QPS reads.", lambda i: {"index":"block_postings","block_target":256,"merge":"batched_vectorized","storage":"lsm_object_storage","writes":"group_commit"}),
  "provenance_acl": ("Check whether derived artifact {i} may influence inference after its source is revoked.", lambda i: {"action":"lifecycle_barrier","check":["origin_key","knowledge_key","generation_key","artifact_key","acl","revocation"],"allow":False}),
  "model_pod": ("Prepare model Pod {i} for adapter activation without changing the base model identity.", lambda i: {"pod_type":"model","action":"activate_adapter","stable_identity":True,"resolve_generation":True,"base_model":"qwen3b","artifact":"lora"}),
  "evaluation": ("Evaluate experiment {i} against the frozen retrieval and generation protocol.", lambda i: {"action":"evaluate","metrics":["recall_at_k","exact_target_match","guardrail_accuracy","latency","qps"],"freeze_protocol":True,"compare_baseline":True}),
}

def make(split):
    rows=[]
    for cap,(template,target_fn) in CAPS.items():
        for i in range(50):
            q=template.format(i=i+1,topic=["similar papers","exact identifier","active Pods","linked facts","lineage","mean of 18 and 24","approved files"][i%7])
            rows.append({"id":f"{split}:curriculum:{cap}:{i}","task":cap,"pod_type":"model" if cap in {"tool_selection","routing_ranker","model_pod"} else "research","language":"en","evidence":None,"question":q,"history":[],"target":json.dumps(target_fn(i),separators=(",",":")),"assessment":{"curriculum_v1":True,"capability":cap,"example_index":i}})
    return rows

for split in ("train","dev","test"):
    p=out/"inputs"/f"{split}.json"; rows=make(split); p.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
cfg=json.loads((out/"inputs/config.json").read_text()); t=cfg["training"]
protocol=json.loads((out/"protocol.json").read_text()); protocol["rows"]={s:len(make(s)) for s in ("train","dev","test")}; protocol["examples_per_capability"]={k:50 for k in CAPS}; protocol["capability_count"]=len(CAPS); protocol["optimizer_updates_per_epoch"]=math.ceil(protocol["rows"]["train"]/t["effective_batch_size"]); protocol["optimizer_updates"]=protocol["optimizer_updates_per_epoch"]*t["epochs"]; protocol["warmup_updates"]=math.ceil(protocol["optimizer_updates"]*t["warmup_ratio"]); protocol["input_hashes"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/"inputs").iterdir()}; protocol["status"]="prepared_not_trained_research_curriculum"; protocol["purpose"]="Research-plan curriculum: Pods, tools, contextual retrieval, chaining, cache, routing, indexes, lifecycle, model Pods, evaluation"; (out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8"); print(protocol["rows"],protocol["optimizer_updates"])
