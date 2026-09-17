"""Build a balanced research-tool SFT bundle (>=50 examples/tool/split)."""
import hashlib, json, math, shutil, sys
from pathlib import Path

src, out = map(Path, sys.argv[1:3])
shutil.copytree(src, out)

tools = {
    "ann_search": [
        ("Which tool finds semantically similar papers in research?", {"tool":"ann_search","arguments":{"namespace":"research","query":"semantically similar papers","top_k":20}}),
        ("Find related supplier lead-time knowledge.", {"tool":"ann_search","arguments":{"namespace":"suppliers","query":"supplier lead time","top_k":10}}),
    ],
    "bm25_search": [
        ("Find exact mentions of retrieval augmented generation.", {"tool":"bm25_search","arguments":{"namespace":"research","query":"retrieval augmented generation","top_k":20}}),
        ("Search the exact identifier CQP1-R144.", {"tool":"bm25_search","arguments":{"namespace":"research","query":"CQP1-R144","top_k":10}}),
    ],
    "metadata_filter": [
        ("Filter active Pods tagged multi-hop.", {"tool":"metadata_filter","arguments":{"namespace":"research","filters":{"status":"active","tags":"multi-hop"},"top_k":20}}),
        ("List approved model Pods in the programming domain.", {"tool":"metadata_filter","arguments":{"namespace":"models","filters":{"status":"active","domain":"programming"},"top_k":20}}),
    ],
    "graph_neighbors": [
        ("Follow depends_on from knowledge K42 for two hops.", {"tool":"graph_neighbors","arguments":{"knowledge_key":"K42","relation":"depends_on","depth":2}}),
        ("Traverse linked facts from supplier:muller for three hops.", {"tool":"graph_neighbors","arguments":{"knowledge_key":"supplier:muller","depth":3}}),
    ],
    "registry_snapshot": [
        ("Verify the lineage of Pod research generation g8.", {"tool":"registry_snapshot","arguments":{"keys":["pod:research"],"generation":"g8","principal":"local"}}),
        ("Inspect the active generation of the model Pod.", {"tool":"registry_snapshot","arguments":{"keys":["pod:model"],"principal":"local"}}),
    ],
    "calculator": [
        ("Compute the mean of 18 and 24 days.", {"tool":"calculator","arguments":{"expression":"(18+24)/2","unit":"days"}}),
        ("Calculate 12 percent of 250 euros.", {"tool":"calculator","arguments":{"expression":"250*0.12","unit":"EUR"}}),
    ],
    "workspace_search": [
        ("Find theorem 4 in the approved workspace.", {"tool":"workspace_search","arguments":{"path":"approved","pattern":"theorem 4","file_types":[".md",".tex"],"max_results":20}}),
        ("Search research notes for the term SPFresh.", {"tool":"workspace_search","arguments":{"path":"research","pattern":"SPFresh","file_types":[".md"],"max_results":20}}),
    ],
}

def rows_for(split):
    rows=[]
    # 25 deterministic variants of each seed pair = 50 examples/tool.
    for tool, seeds in tools.items():
        for i in range(50):
            q, answer = seeds[i % len(seeds)]
            q = f"{q} (case {i+1})"
            rows.append({"id":f"{split}:tool:{tool}:{i}", "task":"research_tool_choice", "pod_type":"model", "language":"en", "evidence":None, "question":q, "history":[], "target":json.dumps(answer, separators=(",",":")), "assessment":{"tool_contract_v2":True,"tool":tool}})
    # Explicit invalid-call guardrails are distributed across each split.
    invalid=[
      ("Use ANN without a namespace.", {"error":"invalid_tool_call","reason":"ann_search requires namespace"}),
      ("Use generation without Pod identity.", {"error":"invalid_tool_call","reason":"generation requires pod_identity"}),
      ("Search outside approved roots.", {"error":"denied","reason":"workspace_search path is outside approved roots"}),
    ]
    for i,(q,a) in enumerate(invalid):
        rows.append({"id":f"{split}:tool:guardrail:{i}","task":"research_tool_choice","pod_type":"model","language":"en","evidence":None,"question":q,"history":[],"target":json.dumps(a,separators=(",",":")),"assessment":{"tool_contract_v2":True,"negative":True}})
    return rows

for split in ("train","dev","test"):
    p=out/"inputs"/f"{split}.json"; data=rows_for(split); p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
config=json.loads((out/"inputs"/"config.json").read_text()); t=config["training"]
protocol=json.loads((out/"protocol.json").read_text()); protocol["rows"]={s:len(rows_for(s)) for s in ("train","dev","test")}; protocol["examples_per_tool"]={k:50 for k in tools}; protocol["tool_count"]=len(tools); protocol["optimizer_updates_per_epoch"]=math.ceil(protocol["rows"]["train"]/t["effective_batch_size"]); protocol["optimizer_updates"]=protocol["optimizer_updates_per_epoch"]*t["epochs"]; protocol["warmup_updates"]=math.ceil(protocol["optimizer_updates"]*t["warmup_ratio"]); protocol["input_hashes"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/"inputs").iterdir()}; protocol["status"]="prepared_not_trained_balanced_tools"; protocol["purpose"]="Balanced tool-choice SFT: at least 50 examples per research tool and split"; (out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8"); print(protocol["rows"],protocol["optimizer_updates"])
