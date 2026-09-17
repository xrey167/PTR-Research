# Dream-RSI integration

Dream-RSI introduces an explicit orchestration layer and uses historical discovery trees as a replay simulator. A policy can be improved offline against realized histories, then deployed online to collect new discoveries and expand the replay pool. The key benefit is lower-cost policy iteration when online evaluations are slow or expensive. [Dream-RSI](https://huggingface.co/papers/2609.14858)

## Fit to our Pod architecture

This becomes a **research-policy Pod**, separate from the Reader and from the document-ranking SID Pod:

```text
discovery traces -> immutable world snapshot -> replay simulator
                 -> offline policy score -> held-out world gate
                 -> online canary -> new discovery tree -> next generation
```

Each world snapshot and tree is content-addressed. A replay result can train or rank a candidate policy, but it cannot by itself activate a generation. Activation requires a held-out world, an online canary, provenance/ACL checks, and rollback to the previous generation.

## Dataset

`runs/dream-rsi-track` contains 50 examples per split with:

- world/tree/policy-generation identifiers;
- candidate actions and historical outcomes;
- decomposed reward: novelty, quality, cost, safety;
- replay-only and online-validation flags;
- anti-leakage checks for future trees and reused test outcomes.

Generator: `research/prepare_dream_rsi_track.py`.

## Training/evaluation order

1. SFT or behavior cloning only for action-schema validity.
2. Offline policy evaluation on replay trees.
3. GRPO/importance-weighted RL using decomposed rewards.
4. Evaluate on unseen world snapshots.
5. Run a small online canary with strict budget and rollback.
6. Promote only if discovery quality improves without cost/safety regression.

This is recursive improvement of the **policy and environment**, not uncontrolled self-editing of the base model.
