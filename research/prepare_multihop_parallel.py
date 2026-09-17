"""Generate balanced Multi-Hop and parallel-search training data."""
import hashlib, json, math, shutil, sys
from pathlib import Path

src, out = map(Path, sys.argv[1:3]); shutil.copytree(src, out)

CAPS = {
  "multihop_reasoning": ("Resolve a {hops}-hop chain from {start} through linked knowledge and return the final value.", lambda i: {"action":"resolve_chain","start":f"knowledge:K{i}","hops":(i%4)+2,"follow_relations":["depends_on","supports"],"return":"final_value"}),
  "multihop_tool_plan": ("Plan retrieval for a question requiring {hops} linked facts.", lambda i: {"action":"iterative_retrieval","hops":(i%4)+2,"steps":["ann_search","graph_neighbors","metadata_filter"],"stop":"all_required_facts_found"}),
  "parallel_fanout": ("Search for independent clues A, B, C, and D in parallel before combining them.", lambda i: {"action":"parallel_search","branches":["clue_a","clue_b","clue_c","clue_d"],"max_concurrency":4,"join":"rank_and_merge"}),
  "hybrid_parallel": ("Run semantic and exact retrieval concurrently for query {i}, then fuse the results.", lambda i: {"action":"parallel_search","branches":[{"tool":"ann_search"},{"tool":"bm25_search"},{"tool":"metadata_filter"}],"join":"rrf","deduplicate":True}),
  "parallel_safety": ("Which plan is valid for query {i}: parallelize independent searches but preserve dependency order?", lambda i: {"action":"schedule","parallel":["independent_searches"],"sequential":["dependent_graph_hops"],"barrier":"all_inputs_ready"}),
  "hop_validation": ("Validate whether the result for chain {i} contains every required hop and provenance link.", lambda i: {"action":"validate_chain","required":["origin_key","knowledge_key","generation_key","artifact_key"],"minimum_hops":2,"reject_incomplete":True}),
  "latency_reward": ("Choose the faster plan for hard search case {i} while preserving recall.", lambda i: {"objective":"maximize_recall_minimize_latency","parallel_calls":min(8,(i%8)+1),"metric":"end_to_end_ms","fallback":"sequential_if_dependency"}),
  "multihop_negative": ("Reject chain {i} because an intermediate fact is missing or revoked.", lambda i: {"action":"reject","reason":"missing_or_revoked_intermediate","allow_inference":False}),
}

def make(split):
 rows=[]
 for cap,(template,fn) in CAPS.items():
  for i in range(50):
   q=template.format(i=i+1,hops=(i%4)+2,start=f"knowledge:K{i+1}")
   rows.append({"id":f"{split}:mhparallel:{cap}:{i}","task":cap,"pod_type":"model","language":"en","evidence":None,"question":q,"history":[],"target":json.dumps(fn(i),separators=(",",":")),"assessment":{"mh_parallel_v1":True,"capability":cap,"example_index":i}})
 return rows

for split in ('train','dev','test'):
 p=out/'inputs'/f'{split}.json'; rows=make(split); p.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
cfg=json.loads((out/'inputs/config.json').read_text()); t=cfg['training']; protocol=json.loads((out/'protocol.json').read_text()); protocol['rows']={s:len(make(s)) for s in ('train','dev','test')}; protocol['examples_per_capability']={k:50 for k in CAPS}; protocol['capability_count']=len(CAPS); protocol['optimizer_updates_per_epoch']=math.ceil(protocol['rows']['train']/t['effective_batch_size']); protocol['optimizer_updates']=protocol['optimizer_updates_per_epoch']*t['epochs']; protocol['warmup_updates']=math.ceil(protocol['optimizer_updates']*t['warmup_ratio']); protocol['input_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'inputs').iterdir()}; protocol['status']='prepared_not_trained_multihop_parallel'; protocol['purpose']='Variable-hop reasoning and parallel search planning curriculum'; (out/'protocol.json').write_text(json.dumps(protocol,indent=2)); print(protocol['rows'],protocol['optimizer_updates'])
