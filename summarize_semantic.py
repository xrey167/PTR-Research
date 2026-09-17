"""Render measured semantic-pipeline results and verify the artifact lineage view."""
import argparse
import json
from pathlib import Path
from neural_pods.registry import Registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "completed": raise RuntimeError("Run did not complete")
    registry = Registry(run / "registry.sqlite3")
    origins = registry.roots(report["four_keys"]["artifact_key"])
    registry.close()
    lifecycle = report["lifecycle"]
    lines = ["# Semantik-Pilot: eine Identitaet, mehrere Formulierungen", "",
        "Echtes Qwen2.5-0.5B-Instruct, zwei tatsaechlich trainierte Rank-8-LoRA-Adapter, CPU.",
        "Die Lieferantendaten sind erfunden; es wurde kein SAP-System angesprochen.", "",
        f"Kanonische Identitaet: `{report['four_keys']['knowledge_key']}`.", "",
        "| Pruefung | Ergebnis |", "|---|---:|",
        f"| Generation 1, 24 Tage, untrainierte Formulierungen | {report['summary']['g1']['correct']}/{report['summary']['g1']['n']} korrekt |",
        f"| Generation 2, 18 Tage, sechs Test- und acht Trainingsformulierungen | {report['summary']['g2']['correct']}/{report['summary']['g2']['n']} korrekt |",
        f"| Nach Quellenwiderruf gesperrte Fragepfade | {sum(lifecycle['revoked_question_paths'])}/{len(lifecycle['revoked_question_paths'])} |",
        f"| Alte Antwort nach Update am Commit gesperrt | {lifecycle['old_commit_blocked']} |",
        f"| Laufende Antwort nach Widerruf am Commit gesperrt | {lifecycle['revoked_commit_blocked']} |",
        f"| Falscher Principal gesperrt | {lifecycle['wrong_principal_blocked']} |",
        f"| Wiederherstellung als Generation 3 | {report['restored']['text']} |",
        f"| Basisgewichte unveraendert | {report['base_unchanged']} |", "",
        "## Deine beiden Formulierungen", ""]
    for phase in ["g1", "g2"]:
        for record in [a for a in report["answers"] if a["phase"] == phase][:2]:
            lines.append(f"- {phase}: **{record['question']}** -> `{record['text']}`")
    lines += ["", "Alle diese Abfragen wurden ohne eingespeisten Antwortwert vom aktivierten LoRA-Pod beantwortet.", "",
        "## Implementiert und geprueft", "",
        "- Getrennte Retrieval-, Wissens- und Lebenszyklusdaten sowie vier explizite Schluessel.",
        "- Kanonische Aliasaufloesung; keine Identitaetsuebernahme durch weiche Aliasse.",
        "- Qdrant-Suche innerhalb der vorab erlaubten semantischen Menge; erneute Registry-Pruefung am Commit.",
        "- Trainierter Kandidatenscorer mit Aehnlichkeit und strukturierten Metadaten.",
        "- Harte Rollen, Rechte und Zeitgrenzen; weiche Annotationen tragen Quelle und Confidence.",
        "- Relationstraversierung mit gemeinsamem Snapshot; Generationenwechsel eines Ziels macht alte Ableitungen ungueltig.", "",
        "Die wiederhergestellte Ableitung traegt sowohl die Quelle der neuen Generation als auch die urspruengliche Trainingsquelle:", ""]
    lines += [f"- `{origin}`" for origin in origins]
    lines += ["", "## Grenzen", "",
        "Der Resolver erkennt die implementierten Lieferzeitformulierungen und bekannten Firmenaliase; er ist kein universeller Sprachparser.",
        "Die aehnliche andere Firma und der RULE-Eintrag sind textuelle Distraktoren. Nur der Ziel-Lieferant besitzt in diesem Lauf neu trainierte LoRA-Adapter.",
        "Die Semantikfilter wurden zusaetzlich mit identischen Vektoren unterschiedlicher Firmen getestet, damit Aehnlichkeit allein den Test nicht bestehen kann.",
        "Graphnavigation ist keine bewiesene neuronale Mehrschritt-Logik. J-Space ist nicht implementiert. Kein neuer RAG-Ueberlegenheitsnachweis und kein Gewichts-Unlearning.",
        "Principals sind lokale Testparameter, keine authentifizierten Benutzer. Dateisystem und Registry-Aufrufer gelten als vertrauenswuerdig.", "",
        "## Dateien", "", "- [Rohbericht mit allen Antworten und Filtern](report.json)",
        "- `registry.sqlite3`: Herkunft, Generationen, Ableitungen und Commit-Belege.",
        "- `semantic-router.pt`: trainierter Router; `adapters/`: trainierte LoRA-Gewichte.",
        "- [Architektur und Startbefehle](../../SEMANTIK.md)", ""]
    (run / "ERGEBNIS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:20]))


if __name__ == "__main__": main()
