# Deadline Status Report

This report is generated offline from local artifacts. It is meant to answer: what can be trusted, what is presentation-ready, and what still needs live API credentials.

## Environment

- Python: `3.11.15` at `/opt/anaconda3/envs/cs224r/bin/python`
- Platform: `macOS-14.4-arm64-arm-64bit`
- Repo `.env`: present
- numpy: OK
- scikit-learn: OK
- pytest: OK
- matplotlib: OK
- sentence-transformers: OK
- faiss-cpu: OK
- tiktoken: OK
- torch: OK
- python-dotenv: OK
- Git status: 1 changed/untracked paths

## Persistent-memory LLM Track

Hand-crafted payload sweep (`results/exp1_summary.json`):

| Stealth | Pairs | Survivors | Retrieval@5 | Judge ASR | Benign ASR |
|---------|-------|-----------|-------------|-----------|------------|
| A overt | 3 | 1/3 | 43% | 33% | 0% |
| B narrative | 4 | 3/4 | 59% | 51% | 0% |
| C indirect | 3 | 1/3 | 38% | 28% | 0% |

Interpretation: retrieval is the first gate; T1/T2 are the weak points, while T3/T4 are already saturated in the existing sweep.

Published-attack baseline (`results/baseline_summary.json`):

| Family | Pairs | Survivors | Retrieval@5 | Judge ASR | Benign ASR |
|--------|-------|-----------|-------------|-----------|------------|
| MEMORYGRAFT | 4 | 4/4 | 70% | 59% | 0% |
| MINJA | 4 | 4/4 | 62% | 62% | 0% |

## Finance Simulator Track

- F0 clean sanity: 32 queries, TruthRetention@5=94%, Tool accuracy=100%

F1 hand-crafted poison by framing style:

| Style | N | IRR@5 | PoisonInTop5 | TruthRetention@5 |
|-------|---|-------|--------------|------------------|
| indirect | 18 | 22% | 78% | 89% |
| narrative | 18 | 44% | 89% | 94% |
| overt | 18 | 44% | 89% | 94% |

Policy baselines:

| Run | Episodes | Mean reward | IRR@5 | Last-50 reward |
|-----|----------|-------------|-------|----------------|
| F2 sparse random | 50 | 0.06 | 6% | 0.06 |
| F3 shaped epsilon-greedy | 50 | 0.46 | 8% | 0.46 |

F4 answer-level defense comparison:

| Mode | N | Uses poison | Contradicts tool | Tool use | IRR@5 |
|------|---|-------------|------------------|----------|-------|
| tool_forced | 18 | 6% | 0% | 83% | 39% |
| tool_optional | 18 | 28% | 22% | 67% | 33% |

## Recommended Close-Deadline Path

1. Use the finance simulator for guaranteed no-API demonstrations: F0 clean sanity, F1 retrieval poisoning, F2/F3 reward-shaping contrast, and F4 tool verification.
2. Treat the committed Exp 1 and published-baseline summaries as the LLM-backed evidence unless you have a stable `.env` and enough time to rerun.
3. If rerunning live LLM experiments, do a one-pair dry run first, then the full sweep. Keep the JSONL logs; the tabulators can regenerate tables without new API calls.

## Commands

Fast offline checks:

```powershell
python scripts/run_offline_finance.py --episodes 500
```

Individual offline commands:

```powershell
python -m pytest finance_poisoning/tests/ -q
python finance_poisoning/experiments/f0_clean_sanity.py
python finance_poisoning/experiments/f1_handcrafted_finance.py --seed 0
python finance_poisoning/experiments/f2_sparse_policy_baseline.py --episodes 500 --seed 0
python finance_poisoning/experiments/f3_shaped_policy_baseline.py --episodes 500 --seed 0
python finance_poisoning/experiments/f4_tool_optional_answer.py --seed 0
python scripts/deadline_report.py
```

Live LLM dry run after `.env` is configured:

```powershell
python scripts/check_modal_ready.py --profile simulation
python scripts/run_live_simulation.py --mode dryrun
python scripts/run_live_simulation.py --mode exp1 --n 20
python scripts/check_modal_ready.py --profile training
python scripts/run_training.py --mode live-exp3 --plot
```

## Things To Be Careful About

- `.env` is currently required for target-agent and judge calls. Without it, stick to finance/offline simulations and existing summaries.
- For Modal, copy `.env.example` to `.env`, replace the Modal host, keep the `/v1` suffix, and use `QWEN_API_KEY=EMPTY` for the default unauthenticated vLLM server.
- The live LLM/RL track needs `sentence-transformers`, `faiss-cpu`, `tiktoken`, and `torch`; the finance simulator does not.
- Do not commit `data/benign_memories.jsonl`; it is a local embedded cache.
- JSONL and PNG outputs are mostly gitignored. Markdown/JSON summaries are the artifacts to keep for the report.
- On this Windows workspace, Git may report dubious ownership. Use a per-command safe-directory override or explicitly mark this repo safe if you trust the path.
- First MiniLM use can download a model. The finance simulator defaults to TF-IDF, so it avoids that dependency unless `FINANCE_RETRIEVER=minilm` is set.
