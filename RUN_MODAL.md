# Running the PISmith GRPO attacker on Modal

Trains a small attacker LLM (default `Qwen2.5-3B-Instruct`, LoRA) with TRL GRPO
so its `<prompt>` memory payloads maximize `PersistentMemoryAttackReward`. The
target victim **and** judge are one deployed `Qwen2.5-7B` vLLM server, reached
through your `.env` `QWEN_BASE_URL`. Only the attacker policy is updated.

## Prerequisites (one-time)

1. **Modal auth:** `pip install modal && modal setup`.
2. **Deploy the target/judge server** (serves Qwen2.5-7B, OpenAI-compatible):
   ```bash
   modal deploy scripts/modal_qwen25_vllm.py
   ```
   It prints a URL like `https://<workspace>--qwen25-memory-redteam-serve.modal.run`.
3. **`.env` at repo root** (already set up on your machine) must point at it:
   ```
   QWEN_API_KEY=EMPTY
   QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
   QWEN_BASE_URL=https://<workspace>--qwen25-memory-redteam-serve.modal.run/v1
   ```
   The trainer injects this `.env` into the GPU container via
   `modal.Secret.from_dotenv()` — no extra secret setup needed. Run the commands
   below from the repo root so `.env` is found.

## Run

```bash
# 1) ~10-min smoke. Acceptance: a `[preflight] judge OK` line (the smoke now
#    validates the judge endpoint), then [PISmith] lines with reward variance > 0
#    (not all-zero) and div_sim < 0.9, and a saved checkpoint. A
#    `[preflight] WARNING: judge call FAILED` means fix the endpoint or run with
#    success_signal=regex before the full run.
modal run scripts/modal_train_grpo.py --smoke

# 2) Full run (~1.5-2.5 h). Checkpoints -> pismith-outputs Volume.
modal run scripts/modal_train_grpo.py

# 3) Optional KL=0 ablation (only after a good baseline run).
modal run scripts/modal_train_grpo.py --beta0
```

## What to watch (printed each step)

```
[PISmith] step=K ASR_retr=.. ASR_judge=.. ASR_regex=.. reward=.. len=..w
          too_short=.. nov_pen=.. div_sim=.. judge_calls=.. judge_err=..
```
- `ASR_retr` should climb first (retrieval is the binding gate), then `ASR_judge`.
- `div_sim` staying < ~0.9 means no monoculture collapse (the novelty penalty working).
- `too_short` near 0 means the min-length floor isn't being gamed.

## Outputs

Checkpoints, the embedded corpus, and per-step metrics persist on the
`pismith-outputs` Modal Volume (`/outputs/pismith_grpo[...]`). For poster plots,
pull `train_metrics.jsonl` (per-step `asr_retr/asr_judge/asr_regex/reward/...`)
and TRL's `trainer_state.json` (reward curve):
```bash
modal volume get pismith-outputs pismith_grpo/train_metrics.jsonl .
modal volume get pismith-outputs pismith_grpo/trainer_state.json .
```

## Tuning knobs

All in `configs/pismith_grpo.yaml`. Common edits:
- `policy_model` — bigger attacker (e.g. `Qwen/Qwen2.5-7B-Instruct`) if the GPU allows.
- `success_signal` — `regex` (fastest), `hybrid` (default), `judge` (most faithful, ~2x calls).
- `max_steps`, `num_generations`, `reward_max_concurrent` — throughput vs. signal.
- If it OOMs: lower `vllm_gpu_memory_utilization`, keep `use_peft: true`, or shrink `policy_model`.
