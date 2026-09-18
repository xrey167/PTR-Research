"""Fictional mutable facts. Question-template families never cross splits."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Fact:
    key: str
    entity: str
    component: str
    value: int
    generation: int = 1

    @property
    def answer(self): return f"{self.value} days"

    @property
    def evidence(self):
        return f"The delivery lead time for {self.component} from {self.entity} is {self.value} days."

    def semantic(self):
        return {"type": "supplier_metric", "subject": self.entity, "predicate": "lead_time",
                "object": {"value": self.value, "unit": "days"}, "component": self.component,
                "tags": ["supplier", "lead-time"], "entities": [self.entity, self.component],
                "confidence": 1.0, "fixture": True}


FACTS = [Fact("supplier:velora:x12:lead_time", "Velora Components", "X12", 24),
         Fact("supplier:norvex:z47:lead_time", "Norvex Supply", "Z47", 37),
         Fact("supplier:talmira:k83:lead_time", "Talmira Industrial", "K83", 16)]

TRAIN = [
    "What is the lead time for {c} from {e}?",
    "How many days does {e} need to deliver {c}?",
    "State the delivery time of {e}'s {c}.",
    "Give the lead time in days for {c}, supplied by {e}.",
    "When ordering {c} from {e}, what is the wait in days?",
    "What delivery delay should I plan for {e} and {c}?",
    "How long is the procurement lead time for {e} {c}?",
    "Tell me the number of days needed for {c} at {e}.",
]
VALIDATION = ["For a purchase of {c}, how long would {e} make us wait?"]
TEST = [
    "Our factory needs {c}. How many days ahead must we order from {e}?",
    "What waiting period applies to shipments of {c} by {e}?",
    "In days, how far in advance should procurement contact {e} about {c}?",
    "Wie viele Tage betraegt die Lieferzeit von {e} fuer {c}?",
]
UNRELATED = [
    ("What is 2 + 3? Reply with only the number.", "5"),
    ("What is the capital of France? Reply with only the city.", "Paris"),
    ("Translate the English word cat into German. Reply with one word.", "Katze"),
]


def questions(fact, split):
    templates = {"train": TRAIN, "validation": VALIDATION, "test": TEST}[split]
    return [t.format(e=fact.entity, c=fact.component) for t in templates]


def dataset_record(facts):
    return {"facts": [asdict(f) for f in facts], "splits": {
        split: [{"key": f.key, "question": q, "answer": f.answer}
                for f in facts for q in questions(f, split)]
        for split in ["train", "validation", "test"]}}
