"""Build a readable evidence report and a reload index from a completed run."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "completed":
        raise RuntimeError("Experiment is not complete")
    # Export the address index for initial runs created before the standalone CLI.
    index = run / "router-manifest.json"
    if not index.exists():
        with sqlite3.connect(run / "registry.sqlite3") as db:
            rows = db.execute("SELECT id,payload FROM nodes WHERE kind='router'").fetchall()
        with (run / "router.pt").open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        candidates = [(key, json.loads(payload)) for key, payload in rows]
        matches = [(key, p) for key, p in candidates if p["payload"]["checkpoint_sha256"] == actual]
        if len(matches) != 1: raise RuntimeError("Could not verify saved router identity")
        dataset = json.loads((run / "dataset.json").read_text(encoding="utf-8"))
        index.write_text(json.dumps({"artifact": matches[0][0], "sha256": actual,
                                    "facts": dataset["facts"]}, indent=2), encoding="utf-8")
    lines = ["# Ergebnis: echtes Qwen-Modell mit versionierten LoRA-Pods", "",
             f"Lauf: `{run.name}`. Geraet: CPU. Laufzeit: {report['elapsed_s']/60:.1f} Minuten.", "",
             "## Unbenutzte Testfragen", "",
             "Drei erfundene Fakten, vier Testformulierungen pro Fakt. Strenges Exact-Match.", "",
             "| Verfahren | Korrekt | Mittlere Latenz | Eingabetokens |",
             "|---|---:|---:|---:|"]
    for mode, row in report["summary"].items():
        lines.append(f"| {mode} | {row['correct']}/{row['n']} | {row['mean_latency_s']:.3f} s | {row['mean_input_tokens']:.1f} |")
    updated = [r for r in report["evaluations"] if r["phase"] == "updated"]
    lifecycle = report["lifecycle"]
    blocked = lifecycle["revoked_paraphrases_blocked"]
    lines += ["", "## Lernen und Lebenszyklus", "",
              f"- Nach Update 24 -> 18 Tage: {sum(r['exact'] for r in updated)}/{len(updated)} Antworten korrekt (Trainings- und Testformulierungen getrennt in JSON nachvollziehbar).",
              f"- Nach Widerruf: {sum(blocked)}/{len(blocked)} gepruefte Fragepfade blockiert.",
              f"- Basisgewichte unveraendert: {report['base_unchanged']}.",
              f"- Adapterdateien insgesamt: {report['adapter_bytes']/1024/1024:.2f} MiB.",
              f"- Wiederhergestellte Generation: {lifecycle['restored_generation']}; Antwort: `{lifecycle['restored_answer']['text']}`.",
              f"- Ungeschuetzter alter Adapter nach Update: `{lifecycle['unguarded_old_adapter_after_update']['text']}`.",
              f"- Mehr-Pod-Komposition: `{report['multi_pod']['result']['text']}`; erwartet `53 days`.", ""]
    for key, value in lifecycle.items():
        if isinstance(value, bool): lines.append(f"- `{key}`: {value}.")
    lines += ["", "## Unerwuenschter Einfluss aktiver Adapter", "",
              "| Adapter | Allgemeine Kontrollfragen korrekt |", "|---|---:|"]
    for adapter in [None, "all_facts", "pod0_g1"]:
        rows = [r for r in report["unrelated"] if r["adapter"] == adapter]
        lines.append(f"| {adapter or 'Basis'} | {sum(r['exact'] for r in rows)}/{len(rows)} |")
    lines += ["", "## Grenzen", "",
              "Kleiner Pilot mit echten vortrainierten Gewichten und echten Gradientenupdates. Kein Nachweis einer Ueberlegenheit gegen Produktions-RAG oder fuer 100–1000 Pods.", "",
              "Die Sperrtests belegen kontrollierten Zugriff und Commit-Konsistenz. Sie belegen kein neuronales Vergessen: Ein direkt aufgerufener alter Adapter kann sein Wissen weiterhin ausgeben.", "",
              "Die Latenzen gelten fuer diesen sequenziellen CPU-Lauf und warme Adapter. Keine Konfidenzintervalle, kein GPU-Test und kein Parallelitaetsbenchmark. Mehr-Pod-Verarbeitung verwendet zwei neural erzeugte Antworten als Text fuer die Komposition; keine Adapterfusion.", "",
              f"Einzel-Pods wurden jeweils {report['config']['steps']} Schritte trainiert, der gemeinsame Adapter {report['config']['steps']*2} Schritte. Gleiche Anzahl von Optimierungsschritten pro Fakt ist damit nicht gegeben; Trainingskosten stehen vollstaendig in report.json.", "",
              "## Belege", "", "- [Rohbericht](report.json)", "- [Trainings- und Testdaten](dataset.json)",
              "- `registry.sqlite3`: unveraenderliche Herkunftsknoten, Elternkanten und Ereignisse.",
              "- `adapters/`: trainierte und wieder geladene Safetensors-Dateien.", "",
              "Die unveraenderte Basismodell-Pruefsumme steht vor und nach dem Training im Rohbericht."]
    verification_path = run / "verification.json"
    if verification_path.exists():
        v = json.loads(verification_path.read_text(encoding="utf-8"))
        if v.get("status") == "completed":
            new = ["## Zusaetzliche Pruefung mit explizit beleggebundenem RAG-Prompt", "",
                   "Frischer Prozess, gespeicherte Adapter, nochmals zwoelf neue Fragen. Keine weiteren Trainingsupdates.", "",
                   "| Verfahren | Exaktes Format | Richtiger Zahlenwert | Mittlere Latenz |",
                   "|---|---:|---:|---:|"]
            for mode, row in v["summary"].items():
                new.append(f"| {mode} | {row['exact']}/{row['n']} | {row['numeric_correct']}/{row['n']} | {row['mean_latency_s']:.3f} s |")
            new += ["", "Der RAG-Prompt wurde nach dem ersten Pilotbefund geaendert, blieb aber erfolglos; die obigen neuen Fragen wurden nicht zum Training verwendet. Der erste schwache RAG-Lauf ist unten vollstaendig erhalten. Auch diese kleine Nachpruefung ist kein Produktionsvergleich.",
                    "", "Rohdaten: [verification.json](verification.json).", ""]
            lines[4:4] = new
    comparison_path = run / "rag_comparison.json"
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        rows = comparison["summary"]
        rag, lookup = rows["test"], rows["structured_lookup"]
        new = ["## Einordnung mit validiertem RAG-Prompt", "",
               f"Drei Promptvarianten wurden ausschliesslich auf den drei reservierten Validierungsfragen verglichen. Gewaehlt: `{comparison['selected_prompt']}`; bei gleicher Zahlenwertgenauigkeit entscheidet die kleinere Eingabelaenge.", "",
               f"- RAG mit gezielter Werteextraktion: **{rag['correct_value']}/{rag['n']} richtige Zahlenwerte** auf einem dritten, neuen Fragensatz; mittlere Latenz {rag['mean_latency_s']:.3f} s.",
               f"- Direkter strukturierter Lookup ohne LLM: **{lookup['correct_value']}/{lookup['n']} richtige Werte**; mittlere Latenz {lookup['mean_latency_s']:.3f} s.", "",
               "Damit ist keine Ueberlegenheit der Pods belegt. Fuer diese strukturierten Einzelwerte funktioniert ein einfacher Lookup bereits sehr gut. Der Extraktionsprompt nutzt die Frage zur Fundstellensuche und laesst das LLM anschliessend nur den Wert aus dem passenden Beleg extrahieren.", "",
               "Die neun Vergleichsfragen unterscheiden sich von den vorigen zwoelf Pod-Fragen; daraus folgt kein statistisch gepaarter Leistungsvergleich. Rohdaten: [rag_comparison.json](rag_comparison.json).", ""]
        lines[4:4] = new
    (run / "ERGEBNIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__": main()
