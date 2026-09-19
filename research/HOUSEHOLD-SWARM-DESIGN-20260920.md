# Household-Swarm-Design: Pods als Colibri/Lumabri-Haushalt — 2026-09-20

Referenzen: JustVugg/colibri (AI-Memory-Multitiering: VRAM/RAM/Storage als
eine Hierarchie, Segment-Residenz, BUSY, Kalibrierung, harte Semantik-
Garantie) und JustVugg/lumabri (Haushalt aus Rechnern: Discovery, Ressourcen-
Angebote, Donor-Approval, Placement-Pläne, BUSY bei Belegung).

**Ziel:** Die neural-pods-Pods verhalten sich mindestens wie diese
Architektur — als Haushalt (Household) aus Pod-Donoren, die Ressourcen
anbieten, Allokationen genehmigen, Modelle in Segment-Plänen verteilen und
calibrierte Geschwindigkeiten melden.

## Die Übersetzung der Verhaltensregeln

| Colibri/Lumabri-Verhalten | neural-pods-Umsetzung |
|---|---|
| LAN-Discovery (UDP 47300) | Mesh-Presence über MQTT (bereits N1: RTT 0,3 ms) |
| Haushalts-Schlüssel + /join | `Household(key)` — Pods joinen per Mesh mit Key-Hash im Envelope |
| „Offer RAM" (/settings) | `offer(pod_id, ram_bytes, vram_bytes)` → Governor-Budget je Pod |
| Donor-Approval vor Start | `AllocationRequest` mit Approvals — Chat startet erst, wenn ALLE Donoren zugestimmt haben |
| Placement-Plan (Layer-Ranges auf Donoren) | `SegmentPlan` — Modell-Segmente (Adapter-Splits, Schichtbereiche) auf Pods verteilt, größenverifiziert |
| `BUSY` bei belegtem Rechner | GovernedExecutorPool/Budget: belegter Pod antwortet BUSY, ersetzt keine laufende Anfrage |
| Kalibrierung (8 Tokens, 20 s Limit) | `calibrate()`: kurze Messprobe je Pod+Modell, Ergebnis gespeichert als „gemessen, nicht kalibriert = keine Speed-Versprechen" |
| VRAM/RAM/Disk als EINE Hierarchie | ResourceGovernor-Tiers (vram→ram→disk Preference-Kette, vorhanden) → Placement wählt Tiers automatisch |
| Harte Semantik-Garantie (kein stilles Precision-Ändern) | Gate-Prinzip: Quantisierung/Präzision steht im Pod-Manifest, Änderung = neue Generation, nie still |

## Umsetzung: `neural_pods/household.py`

```python
class Household:
    join(pod_id, key, offered_ram, offered_vram)   # Donor-Angebot
    members() -> [offers]                           # Inventar (angebote + busy)
    request_allocation(model_spec) -> AllocationPlan
        # SegmentPlan: Segmente auf Donoren verteilt, größenverifiziert,
        # mit Approval-Liste; startbar erst wenn alle approved
    approve(request_id, pod_id)                     # Donor-Zustimmung
    start(plan) -> reservation                      # BUSY wenn belegt
    calibrate(pod_id, model_ref) -> Calibration     # kurze Messprobe
```

- **Segment:** abstrakt — ein Adapter-Shard, ein Schichtbereich oder ein
  Reranker; das Verhalten (Residenz je Donor, Chat-Host getrennt) ist der
  Vertrag, nicht das Segment-Format
- **Approval:** Provenance-gebunden — jede Zustimmung ist ein Registry-Event
  (wer, wann, für welche Plan-Hash) — auditierbar wie im Haushalt
- **BUSY:** ist ein Zustand am Donor (Residency-Lease aktiv), keine Fehler-
  Abweisung — der Plan kann den Donor für spätere Segmente wieder einplanen

## Phasen

| Phase | Inhalt | Gate-Check |
|---|---|---|
| H1 | Household: join/offer/members + Key-Bindung, BUSY-Semantik | `household_join` |
| H2 | request_allocation → SegmentPlan (größenverifiziert, Approval-Pflicht, BUSY-Respekt) | `segment_plan` |
| H3 | Calibration (Messprobe je Pod+Modell, gespeichert) | `household_calibration` |
| H4 | Live-Demo über Mesh: 3 Pod-Donoren (Host + 2 LXD), Plan → Approvals → Reservierung → Segment-Residenz | `household_e2e` |

Akzeptanz: kein bestehender Check bricht; der Haushalt lehnt Überbelegung
ab, startet nie ohne volle Approvals, und jede Kalibrierung ist als
Messdatei nachvollziehbar.


## Erweiterung: Training, Lernen und Token-Cache im Haushalt

| Verhalten | Umsetzung |
|---|---|
| Donoren stellen Trainings-Rechenleistung | `request_training(dataset_ref, budget)` — der Dream-Pod/Improve-Kreislauf läuft auf Haushalts-Donoren; Approval gilt auch hier (GPU-Donor approved den Trainingslauf); Autonomie-Budget aus dem Pod-Arm-Design zählt pro Donor |
| Lernen = Generationen-Zyklus | Trainings-Outcomes (frozen A/B) fließen in den Dream-Pool — der Haushalt „lernt" als Einheit: neue Generation wird im Haushalt per Gate promoviert und alle Pods ziehen sie per Registry-Supersedes |
| Token-Cache (KV/Prefix, Mooncake/tair) | `TokenCache`: Pod-übergreifender Prefix-/Token-Cache in Redis (L1) — session_id → {replica, prefix_hash, token_count}; ein Pod, der denselben Prompt-Präfix bedient, erbt die gecachten Token; Metrik: cache_hits, saved_tokens |
| Token-Fluss zwischen Pods | Task-Graph-Kanten (N3) tragen Token-Kontext; der TokenCache speichert wiederverwendbare Präfixe je (session, prefix_hash) — Affinity hält die Session auf der Replica mit warmem KV |

Zusatz-Phasen:

| Phase | Inhalt | Gate-Check |
|---|---|---|
| H5 | Training im Haushalt: request_training mit Donor-Approval + Autonomie-Budget, Dream-Pod-Kopplung | `household_training` |
| H6 | TokenCache: Prefix-/Session-Affinity über Pods, saved_tokens-Metrik, Failover-Verhalten | `token_cache` |
