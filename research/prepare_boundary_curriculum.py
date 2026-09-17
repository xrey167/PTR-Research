"""Generate unseen arithmetic boundary examples for reader continuation."""
from __future__ import annotations
import json
from pathlib import Path

def _weeks(n: int, language: str) -> str:
    w, d = divmod(n, 7)
    if language == "de": return f"{w} Wochen" + (f" und {d} Tage" if d else "")
    return f"{w} weeks" + (f" and {d} days" if d else "")

def rows(split: str = "train") -> list[dict]:
    if split != "train": return []
    suppliers = [("Nordstern Parts", "N41"), ("Rheinland Motion", "R52"), ("Atlas Mechatronics", "A63"), ("Komet Systems", "K74")]
    out=[]
    for si, (supplier, component) in enumerate(suppliers):
        lead = 35 + si * 7
        for language in ("en", "de"):
            for deadline in (lead - 7, lead - 1, lead, lead + 1, lead + 7):
                sufficient = deadline >= lead
                if language == "en":
                    question = f"For {supplier} / {component}, is a deadline of {_weeks(deadline, language)} sufficient?"
                    target = f"{'Yes' if sufficient else 'No'}; {lead} days"
                else:
                    question = f"Sind {_weeks(deadline, language)} für {supplier} / {component} ausreichend?"
                    target = f"{'Ja' if sufficient else 'Nein'}; {lead} Tage"
                out.append({"id": f"boundary:{component}:{lead}:{language}:{deadline}", "task":"deadline_weeks_boundary", "pod_type":"math", "language":language,
                    "evidence":{"supplier":supplier,"component":component,"lead_time_days":lead}, "question":question, "history":[], "target":target,
                    "assessment":{"supplier":supplier,"component":component,"value":lead,"deadline_days":deadline,"answer":sufficient}})
            for value in (lead - 1, lead, lead + 1):
                if language == "en": question=f"For {supplier} / {component}, what duration includes four extra buffer days?"; target=f"{value + 4} days"
                else: question=f"Welche Dauer ergibt sich für {supplier} / {component} mit vier Puffertagen?"; target=f"{value + 4} Tage"
                out.append({"id":f"boundary:{component}:{value}:{language}:buffer","task":"followup_buffer","pod_type":"math","language":language,
                    "evidence":{"supplier":supplier,"component":component,"lead_time_days":value},"question":question,"history":[],"target":target,
                    "assessment":{"supplier":supplier,"component":component,"value":value,"deadline_days":None,"answer":value+4}})
    return out

if __name__ == "__main__":
    p=Path("runs/boundary-curriculum-001"); p.mkdir(parents=True,exist_ok=True); rs=rows(); (p/"train.json").write_text(json.dumps(rs,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); print(json.dumps({"rows":len(rs),"path":str(p/"train.json")}))
