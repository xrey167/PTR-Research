"""Generate balanced Pod-type/domain routing data with held-out templates."""
from __future__ import annotations
import json
from pathlib import Path

TEMPLATES = {
    "context": ["Which entity is linked to {item}?", "Summarize the background of {item}.", "What does the record say about {item}?"],
    "math": ["Calculate the value for {item}.", "How many units are associated with {item}?", "Compute the percentage change for {item}."],
    "model": ["Which policy should govern {item}?", "Decide whether the rule applies to {item}.", "What action does the model recommend for {item}?"],
    "reasoning": ["Trace the evidence chain connecting {item}.", "Infer the cause across the records for {item}.", "Which multi-hop conclusion follows from {item}?"],
    "retrieval": ["Find the supporting document for {item}.", "Retrieve matching evidence about {item}.", "Search aliases and records for {item}."],
}
HELDOUT_TEMPLATES = {
    "context": ["Give the background associated with {item}.", "Which record describes {item}?"],
    "math": ["What is the numerical result for {item}?", "Determine the quantity for {item}."],
    "model": ["Select the governing policy for {item}.", "Which decision should be made about {item}?"],
    "reasoning": ["Combine the linked evidence to explain {item}.", "What conclusion follows through the chain for {item}?"],
    "retrieval": ["Locate evidence that supports {item}.", "Which record should be fetched for {item}?"],
}
DOMAINS = ["business", "economics", "law", "science", "engineering"]
ROLES = {"context":"FACT", "math":"CALCULATION", "model":"RULE", "reasoning":"RELATION", "retrieval":"DOCUMENT"}
INTENTS = {"context":"lookup_context", "math":"compute_value", "model":"apply_policy", "reasoning":"multi_hop_inference", "retrieval":"find_evidence"}
TAGS = {"context":["entity","background"], "math":["numeric","units"], "model":["policy","decision"], "reasoning":["multi_hop","evidence_chain"], "retrieval":["search","alias"]}


def build(per_type=100):
    rows=[]; items=["supplier lead time", "bond price", "approval clause", "sensor failure", "engine tolerance"]
    heldout_items=["vendor transit interval", "interest-rate valuation", "compliance exception", "reactor anomaly", "machine fit margin"]
    for type_index, (pod_type, templates) in enumerate(TEMPLATES.items()):
        for i in range(per_type):
            domain=DOMAINS[i % len(DOMAINS)]; item=items[i % len(items)]
            split="train" if i < int(per_type * 0.6) else "validation" if i < int(per_type * 0.8) else "test"
            if split == "train":
                template=templates[i % len(templates)]; chosen_item=item
            else:
                template=HELDOUT_TEMPLATES[pod_type][i % len(HELDOUT_TEMPLATES[pod_type])]; chosen_item=heldout_items[i % len(heldout_items)]
            rows.append({"id":f"{pod_type}-{i:03d}","question":template.format(item=chosen_item),"pod_type":pod_type,"domain":domain,"semantic_role":ROLES[pod_type],"intent":INTENTS[pod_type],"tags":TAGS[pod_type],"split":split})
    return rows


def render(per_type: int = 100) -> str:
    """The dataset as it is written to disk. Deterministic: same input, same
    bytes, so the checked-in artifact can be compared against the code."""
    return "".join(json.dumps(r) + "\n" for r in build(per_type))


# Default output lives under research/runs/, which .gitignore keeps (see the
# !research/runs/ exception). The dataset is pure code output, so shipping it
# with the repository is what lets a clone run the routing tests at all.
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "runs" / "taxonomy-routing-balanced-003.jsonl"


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows":len(rows),"per_type":{t:sum(r["pod_type"]==t for r in rows) for t in TEMPLATES},"splits":{s:sum(r["split"]==s for r in rows) for s in ("train","validation","test")}},indent=2))
