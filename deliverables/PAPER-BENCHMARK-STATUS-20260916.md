# Paper benchmark status — 2026-09-16

## What is prepared

The project contains a pinned, reproducible manifest for the two comparison cohorts used by the referenced search research:

- OSWorld-V2 `v2026.08.08`: 108 tasks
- Agents' Last Exam near-term cohort `d10fb61a14f9719774c3520c5763068b28ef554d`: 67 tasks (64 CPU + 3 GPU baselines)

Both baseline and RSI arms are specified. Evaluation keeps the verifier outside learning, freezes memory during test, and records partial score, success, wall time, tool calls, input tokens, and memory Pods.

## What was actually measured

The local system and Qwen LoRA were measured on `xrserver-dev` (RTX 3090). The persisted GPU generation benchmark reports 22.48 generated tokens/s and 622.87 ms batch latency. These are engineering measurements for our stack, not OSWorld/ALE scores.

## Why official scores are not claimed yet

The official VM runners are not executable in the current environment. `xrserver-dev` has `/dev/kvm` and a writable `/srv/ai/workspaces` mount with 154.96 GiB free, but Docker is absent. The bundled LXD QEMU binary is not standalone-ready because `libspice-server.so.1` is unavailable. OSWorld task assets and the model credentials are also not provisioned.

The dry-run therefore remains `prepared_not_run` and deliberately emits no fabricated score. Once Docker/QEMU and the official assets are provisioned, the manifest can be handed to the pinned runners without changing the evaluation protocol.

Evidence: `runs/paper-benchmark-manifest-001.json` (local), `runs/paper-benchmark-manifest-xrserver.json` (remote audit), and `runs/qwen-gpu-xrserver-001.json`.
