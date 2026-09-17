# Neural Pods demo

Run from `C:\Users\ReyDa\neural-pods`:

```powershell
.venv\Scripts\python.exe -m research.demo_concept_pod
```

The command creates `runs/concept-pod-demo-001.json`. It shows one
provenance-marked Pod being activated in a tiny Qwen decoder, changing the
next-token logits, and then being revoked. Revocation masks the concept and
restores the baseline output exactly.

This is a systems demonstration, not a language-quality benchmark. For a
public presentation, show the JSON report alongside the architecture diagram,
then run the same flow with a real checkpoint after access and privacy review.
The real Qwen3B lifecycle precision reports are in
`runs/qwen-lifecycle-gate-3b-*.json`.

## Pod gallery

```powershell
.venv\Scripts\python.exe -m research.demo_pod_gallery
```

This runs six questions through context, math and model Pods. The report
records which Pod type was selected, the active logit change, a wrong-type
control, and the revocation result. The routing in this small demo is a
deterministic scaffold; learned Dragonfly routing is evaluated separately.
