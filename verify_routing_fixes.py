import json, sqlite3, tempfile, shutil
from pathlib import Path
from sentence_transformers import SentenceTransformer
from neural_pods.registry import Registry, InvalidState, verify_files
from neural_pods.semantic_routing import SemanticRouter
from neural_pods.model import PodModel

run=Path("runs/semantic-003").resolve()
report=json.loads((run/"report.json").read_text(encoding="utf-8"))
result={"isolation":"temporary copy of saved registry and Qdrant; original model and adapter files read only","cases":[]}
with tempfile.TemporaryDirectory(prefix="pods-model-review-") as tmp:
    tmp=Path(tmp)
    source=sqlite3.connect((run/"registry.sqlite3").as_uri()+"?mode=ro",uri=True)
    dest=sqlite3.connect(tmp/"registry.sqlite3")
    source.backup(dest); dest.close(); source.close()
    shutil.copytree(run/"qdrant",tmp/"qdrant")
    for name in ["semantic-router.json","semantic-router.pt"]:
        shutil.copyfile(run/name,tmp/name)
    registry=Registry(tmp/"registry.sqlite3")
    encoder=SentenceTransformer(report["models"]["encoder"]["path"],device="cpu",local_files_only=True)
    router=SemanticRouter(registry,encoder,tmp); router.load_weights()
    model=PodModel(report["models"]["qwen"]["path"])
    assert model.frozen_hash()==report["base_weights_sha256"]
    loaded=set()
    cases=[
        ("control","What is the delivery lead time for X12 from Mueller GmbH?","known X12 fact"),
        ("unknown_component","What is the delivery lead time for X99 from Mueller GmbH?","reject or defer: no X99 fact exists"),
        ("wrong_intent","How long has Mueller GmbH existed?","reject or defer: company age is not delivery lead time"),
    ]
    for name,question,expected in cases:
        case={"case":name,"question":question,"expected":expected}
        try:
            selected=router.select(question,principal="buyer")
            payload=registry.node(selected["artifact_key"])["payload"]["payload"]
            adapter=payload["adapter"]
            if adapter not in loaded:
                path=run/"adapters"/adapter
                verify_files(path,payload["files"])
                model.model.load_adapter(path,adapter_name=adapter,is_trainable=False); loaded.add(adapter)
            output=model.generate(question,adapter=adapter)
            receipt=registry.commit(selected["snapshot"],output["text"])
            case.update(status="committed",output=output["text"],selected_key=selected["knowledge_key"],
                        parsed=selected["query_semantics"],score=selected["score"],answer_id=receipt["answer_id"])
        except InvalidState as exc:
            case.update(status="blocked",reason=str(exc))
        result["cases"].append(case)
        print(json.dumps(case),flush=True)
    from run_semantic_experiment import HELD_OUT
    for question in HELD_OUT:
        selected=router.select(question,principal="buyer")
        output=model.generate(question,adapter=adapter)
        assert output["text"] == "24 days", (question, output)
        registry.commit(selected["snapshot"],output["text"])
        result["cases"].append({"case":"held_out","question":question,"status":"committed","output":output["text"]})
    assert result["cases"][0]["output"] == "24 days"
    assert all(c["status"] == "blocked" for c in result["cases"][1:3])
    result["status"]="passed"
    router.close(); registry.close()
Path("runs/fix-model-probes.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
