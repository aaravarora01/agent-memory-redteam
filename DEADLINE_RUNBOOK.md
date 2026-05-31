# Deadline Runbook

This is the shortest reliable path for running and explaining the project under
deadline pressure. It separates the no-API finance simulator from the live LLM
persistent-memory experiments, so you can still produce interpretable results
even if credentials or network access are shaky.

## 1. Fast Path: No API Required

Run this from the repository root:

```powershell
python scripts/run_offline_finance.py --episodes 500
```

That command runs tests, clean finance sanity, hand-crafted finance poisoning,
sparse/shaped finance training baselines, tool-verification comparison, and the
deadline report.

Outputs to look at:

- `finance_poisoning/results/f1_by_style.md`: retrieval poisoning by framing style.
- `finance_poisoning/results/tool_optional_summary.json`: tool-optional vs tool-forced answer corruption.
- `finance_poisoning/results/f2_reward_curve.png`: sparse random baseline curve.
- `finance_poisoning/results/f3_reward_curve.png`: shaped epsilon-greedy baseline curve.
- `results/deadline_report.md`: one-page status across the repo.

## 2. Live LLM Simulation

Create a repo-root `.env` first. For Modal + vLLM Qwen2.5, copy
`.env.example` to `.env` and replace the URL:

```text
QWEN_API_KEY=EMPTY
QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
QWEN_BASE_URL=https://your-workspace--qwen25-memory-redteam-serve.modal.run/v1
```

For hosted DashScope/Qwen:

```text
QWEN_API_KEY=...
QWEN_MODEL=qwen-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Check readiness before making live calls:

```powershell
python scripts/check_modal_ready.py --profile simulation
```

Then run the smallest live simulation:

```powershell
python scripts/run_live_simulation.py --mode dryrun
```

After the dry run passes:

```powershell
python scripts/run_live_simulation.py --mode smoke --seeds 0 1 2 3 4
python scripts/run_live_simulation.py --mode exp1 --n 20
python scripts/run_live_simulation.py --mode baseline --n 20
```

## 3. Training

No-API finance training baselines:

```powershell
python scripts/run_training.py --mode finance --episodes 500
```

Live sparse-PPO training:

```powershell
# Cheap training smoke, 24 live target-agent calls
python scripts/check_modal_ready.py --profile training
python scripts/run_training.py --mode live-exp3 --plot

# Full Exp 3 sweep
python scripts/run_training.py --mode live-exp3 --episodes 2000 --batch-size 64 --minibatch-size 64 --ppo-epochs 4 --wall-clock-cap 7200 --plot
```

## 4. Interpretation Checklist

- The core novelty is the temporal gap: poison memory now, retrieve and execute later.
- Attack success factors into retrieval and execution. If the poison is not in top-k, the agent usually never sees it.
- Exp 1 is the hand-crafted floor. Published single-payload ports are the ceiling.
- Finance experiments are a controlled synthetic benchmark: no real APIs, no real money movement, deterministic tools, and interpretable metrics.
- The finance simulator cleanly shows the defense story: read-only tools help, and forced verification is stronger than optional verification.

## 5. Risk Checklist

- Missing `.env`: live LLM scripts fail with `No chat model API key is set`.
- Wrong Python environment: prefer Python 3.11 for the live LLM/RL track.
- Live LLM/RL needs `sentence-transformers`, `faiss-cpu`, `tiktoken`, and `torch`. Finance offline does not.
- Modal URLs should end in `/v1`; keep `QWEN_API_KEY=EMPTY` for the default unauthenticated vLLM server.
- First Modal request may cold-start. The launchers default to `--request-timeout 300`.
- Start with dry runs before `--n 20` sweeps or 2000-episode training.
- Keep JSONL logs. Summaries/tables/plots can be regenerated offline.
- Do not commit `.env` or `data/benign_memories.jsonl`.
- If results look bad, inspect retrieval first: if the payload is not in top-k, execution cannot happen.
