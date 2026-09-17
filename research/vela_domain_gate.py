"""Advisory Vela domain gate; hard lifecycle checks remain downstream."""
from transformers import pipeline


def load(model_id="llm-semantic-router/Vela-1.0-Encoder-307M-Domain", device=-1):
    return pipeline("text-classification", model=model_id, device=device)


def classify(gate, question, threshold=0.8):
    scores = gate(question, top_k=None, truncation=False)
    top = max(scores, key=lambda item: item["score"])
    return {"label": top["label"], "score": float(top["score"]),
            "advisory": True, "confident": bool(top["score"] >= threshold)}
