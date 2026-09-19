# Typed Pods and model behavior

One common lineage envelope does not mean every Pod should have the same
payload contract. The implementation now distinguishes three types:

| Type | Payload | Gate before model use |
|---|---|---|
| `context` | prose/evidence, aliases, relations | provenance, ACL, novelty and contradiction thresholds |
| `math` | operands, expression, unit, rounding | deterministic typed evaluation; generated prose cannot change the result |
| `model` | variant (`lora`, `distilled`, `moe`, `quantized`, `base`), model/adapter hash, input/output schema | exact artifact identity, variant metadata and executable contract |

All three still carry OriginKey, KnowledgeKey, GenerationKey and ArtifactKey.
For derived or model Pods with several parents, the header retains
`origin_keys[]`, `knowledge_keys[]` and `generation_keys[]` instead of dropping
lineage to one arbitrary parent.
The distinction controls retrieval and execution, not lineage. A context Pod
may be retrieved into a reasoning window, a math Pod is routed to a calculator
guard, and a model Pod is activated only after its adapter identity is checked.

`research/reader_training_data.py` labels the reader curriculum with
`pod_type`; the prompt explains the contracts and includes model-Pod examples.
This teaches the Qwen reader which operation is allowed before it sees the
answer. The deterministic guard remains the final safety layer for arithmetic.

Model Pods are a family of executable artifacts, not a synonym for LoRA. A
distilled Pod binds `teacher_identity`; a MoE Pod binds an expert manifest;
quantized and base variants bind a model hash. Every variant still passes the
same generation, ACL and revocation barrier.

`neural_pods.model_pod.ModelPodRuntime` resolves the validated artifact into a
server-neutral activation plan (`attach_lora`, `load_student`, `load_moe`,
`load_quantized` or `load_base`). A serving backend can execute that plan, but
cannot select a different generation after the manifest has been validated.
