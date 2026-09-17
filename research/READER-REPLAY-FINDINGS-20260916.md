# Reader replay findings (2026-09-16)

The mixed Qwen/PEFT reader remains the selected adapter:

| adapter | two-hop holdout | three-hop holdout | optimizer steps |
|---|---:|---:|---:|
| `reader-lora-mixed-multihop-001` | 84/96 | 4/4 | 32 |
| `reader-lora-fullreplay-001` | 64/96 | 3/4 | 760 |

The second run replayed all 372 original training rows plus eight new
three-hop rows for two epochs. It is retained as a negative ablation and is
not registered for serving. Its regression is evidence that a large,
unweighted replay pass overwrites short-answer formatting and boundary
comparisons. Future continuation runs should use balanced sampling and a
lower learning rate, with the frozen 96-question holdout checked after every
epoch.

The 12 failures of the selected adapter are localized to four held-out
boundary cases repeated across two entities and two languages:

- eight `deadline_weeks_short` cases (52/56 days versus an eight-week limit),
- four `followup_buffer` cases (56 days plus four days should be 60).

The core two-hop and all three-hop cases pass. This report keeps the failed
ablation visible so it cannot be mistaken for an improvement.

