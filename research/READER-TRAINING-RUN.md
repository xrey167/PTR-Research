# Reader training entry point

`train_reader.py` implements the fixed candidate in
`runs/reader-training-inputs-001`. It is separate from the trained address planner
and the learned semantic links. A completed training run is explicitly reported
as `trained_not_evaluated`; it is not an integrated neural Pod yet.

## Execution order

Do not run another 3B model concurrently with the active BF16 integration run on
this 15.65 GiB Windows host. After that process is terminal, the reader deadline
diagnosis remains useful before selecting a training experiment.

Run the memory probe using a new output directory:

```powershell
.\.venv\Scripts\python.exe -X utf8 research/train_reader.py preflight --inputs runs/reader-training-inputs-001 --output runs/reader-training-preflight-001
```

This verifies the frozen files and actual base model hashes, loads BF16 Qwen3B,
adds LoRA to attention and MLP projections, and runs two complete accumulation
windows using the longest training sequence. It exercises backward computation,
gradient clipping, Adam state allocation and a second update with that state
resident. By default it uses one longest-row microbatch for one bounded
optimizer step (`--probe-microbatches 1`); increase that only on a host where a
longer probe is acceptable. The probe adapter is discarded. Process working set, Windows peak
working set/private bytes, available memory and step timings are recorded.
Successful execution alone does not imply acceptable speed or stable future
memory availability; inspect those measurements before the full run.

On the current Windows host the real probe loaded Qwen3B and entered its
first BF16 backward pass. At the latest observation it remains responsive at
about 7.93 GB private memory, but no optimizer step has completed after roughly
13 minutes of CPU paging. This is an observed host limitation, not a successful
training result. That process was stopped after more than 20 minutes without an
optimizer step and is recorded in `runs/reader-training-preflight-001/host-limit.json`.
The shorter bounded probe must be run separately before full training.

The trainer now also runs a daemon progress monitor during long backward passes,
writing elapsed time, resident memory and available system memory every 30
seconds. A process loaded before this change cannot use the new monitor; its
live process state remains authoritative.

Once the probe is reviewed, the matching training command is:

```powershell
.\.venv\Scripts\python.exe -X utf8 research/train_reader.py train --inputs runs/reader-training-inputs-001 --preflight runs/reader-training-preflight-001 --output runs/reader-lora-001
```

The trainer checks the probe's completed state, device, frozen protocol and
runner hash. It starts a fresh adapter from the fixed seed, runs two epochs with
deterministic shuffling, 46 optimizer updates and a token-normalized partial
window at each epoch end. It verifies unchanged frozen parameters and changed
trainable parameters. The saved reader identity includes effective LoRA settings
and the attention implementation. Its adapter files are hashed.

## Validation and remaining work

The tiny real Qwen/PEFT test exercises four updates across two epochs, partial
windows, the learning-rate schedule, frozen/adapter hashes and adapter save/load.
Both sides of its exact output comparison must explicitly use eager attention;
automatic attention selection after reload can produce numerical differences.
This test is not evidence that the 3B model has trained or learned the task.

The 3B probe and full training have not yet run. No development or final test
labels are consumed by this trainer. Separate real generation evaluations must
compare the same held-out inputs against the base and adapted readers. Exact
target matching is only a format metric; review semantic answers as well.
Afterward, register model/training provenance, compile fresh identity-bound
capsules and rerun dialogue, lifecycle and revocation checks. Old KV capsules
must not be reused with the trained reader. Broader research requirements remain
open, including multihop reasoning and original compact neural-state machinery.

## Frozen held-out evaluation entry point

`evaluate_reader.py` now implements separate base/adapter evaluations. For example:

```powershell
.\.venv\Scripts\python.exe -X utf8 research/evaluate_reader.py --inputs runs/reader-training-inputs-001 --split dev --output runs/reader-base-dev-001
.\.venv\Scripts\python.exe -X utf8 research/evaluate_reader.py --inputs runs/reader-training-inputs-001 --split dev --adapter-run runs/reader-lora-001 --output runs/reader-adapter-dev-001
```

Run these sequentially after the memory-intensive current job. Neither command
has been run with 3B yet. Use `--split test` in distinct output directories for
the fixed final evaluation, without using test answers for checkpoint selection.
The evaluator verifies input/model/adapter hashes, enforces the trained reader's
full identity after reload, and saves actual output tokens, first-token logits,
prompt segments, timings and the unchanged final reader identity. Both variants
use eager attention and BF16. Existing SDPA control results are not the paired
baseline for this new experiment. It reports exact-match counts explicitly as
format scores, not semantic correctness. No target or assessment field is
passed into prompt rendering.

## Reader-bound capsules

`research/reader_capsule.py` adds `ReaderCapsules`, compatible with the
`LinkedCapsules` interface. It requires a registered `lora` artifact with schema
`research-reader:v1` and the full `reader_identity` payload. That artifact must
have the ordinary canonical training-procedure and source ancestry required by
the registry. Every compiled KV artifact directly depends on both the current
fact generation and this reader artifact. Reader/training revocation therefore
propagates to its capsules and answer materializations, without revoking an
independent supplier fact.

The implementation checks actual weights and effective adapter configuration
before loading/decoding and after compilation/decoding. Changing LoRA scaling
invalidates its identity even if all weight tensors are unchanged. The full hash
checks are intentionally a research correctness path: their cost on 3B has not
been measured and they do not establish low lifecycle overhead. The current
large BF16 dialogue job still runs its preserved original implementation.

This new path uses a manually populated tiny LoRA fixture for its focused test;
the fixture is not a trained reader or a semantic-quality result. Registering a
real completed 3B training run and exercising it through learned routing remain
pending. The payload is still full-prefix KV, not the recovered CQP1/J-Space
compact-state architecture.
