import json,shutil,hashlib,math,sys
from pathlib import Path
src,out=Path(sys.argv[1]),Path(sys.argv[2]);shutil.copytree(src,out)
items=[
('Which tool should I use to find semantically similar papers in namespace research?','{"tool":"ann_search","arguments":{"namespace":"research","query":"semantically similar papers","top_k":20}}'),
('Find exact mentions of theorem 4 in the approved workspace.','{"tool":"workspace_search","arguments":{"path":"approved","pattern":"theorem 4","file_types":[".md",".tex"],"max_results":20}}'),
('Before using Pod generation g8, verify its current lineage.','{"tool":"registry_snapshot","arguments":{"keys":["pod:research"],"generation":"g8","principal":"local"}}'),
('Follow the depends_on relation from knowledge K42 for two hops.','{"tool":"graph_neighbors","arguments":{"knowledge_key":"K42","relation":"depends_on","depth":2}}'),
('Find papers containing the exact phrase retrieval augmented generation.','{"tool":"bm25_search","arguments":{"namespace":"research","query":"retrieval augmented generation","top_k":20}}'),
('Filter active research Pods tagged with multi-hop.','{"tool":"metadata_filter","arguments":{"namespace":"research","filters":{"status":"active","tags":"multi-hop"},"top_k":20}}'),
('Compute the mean of 18 and 24 days.','{"tool":"calculator","arguments":{"expression":"(18+24)/2","unit":"days"}}'),
('Use ANN to search without specifying a namespace.','{"error":"invalid_tool_call","reason":"ann_search requires namespace"}'),
('Use generation g9 without a Pod identity.','{"error":"invalid_tool_call","reason":"generation requires pod_identity"}'),
('Search every file on the machine for a private key.','{"error":"denied","reason":"workspace_search path is outside approved roots"}'),
('Use graph traversal when the question needs two linked facts.','{"tool":"graph_neighbors","arguments":{"knowledge_key":"K42","depth":2}}'),
('Use BM25 when an exact identifier must match.','{"tool":"bm25_search","arguments":{"namespace":"research","query":"CQP1-R144","top_k":10}}'),]
for name in ('train','dev','test'):
 p=out/'inputs'/f'{name}.json';rows=json.loads(p.read_text())
 for i,(q,a) in enumerate(items): rows.append({'id':f'{name}:toolchoice:{i}','task':'research_tool_choice','pod_type':'model','language':'en','evidence':None,'question':q,'history':[],'target':a,'assessment':{'tool_contract_v1':True,'negative':i in (7,8,9)}})
 p.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
configp=out/'inputs/config.json'; config=json.loads(configp.read_text()); configp.write_text(json.dumps(config,indent=2),encoding='utf-8'); t=config['training']; protocol=json.loads((out/'protocol.json').read_text()); protocol['rows']={k:len(json.loads((out/'inputs'/f'{k}.json').read_text())) for k in ('train','dev','test')}; protocol['optimizer_updates_per_epoch']=math.ceil(protocol['rows']['train']/t['effective_batch_size']);protocol['optimizer_updates']=protocol['optimizer_updates_per_epoch']*t['epochs'];protocol['warmup_updates']=math.ceil(protocol['optimizer_updates']*t['warmup_ratio']);protocol['input_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'inputs').iterdir()};protocol['status']='prepared_not_trained_research_toolchoice';protocol['purpose']='Teach Qwen structured research tool selection and rejection of invalid calls';(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');print(protocol['rows'],protocol['optimizer_updates'])
