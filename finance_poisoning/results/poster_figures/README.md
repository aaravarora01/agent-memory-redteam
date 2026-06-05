# Poster Figure Pack

Generated from:
- `stage1_train_metrics.jsonl`
- `stage2_train_metrics.jsonl`
- `finance_grpo_stage2_judge_shaped_checkpoint-300_eval.jsonl`
- `finance_grpo_stage2_judge_shaped_checkpoint-300_eval_summary.json`

Figures:
1. `training_reward_curve.png` — Stage 1/Stage 2 reward curve with moving average.
2. `training_component_curves.png` — valid action, retrieval@5, scorer ASR, judge ASR over training.
3. `stage_mean_bars.png` — Stage 1 vs Stage 2 aggregate metrics.
4. `eval_by_fact.png` — final eval metrics by target fact.
5. `eval_funnel.png` — final eval pipeline funnel: generated → valid → retrieved → scorer → judge.
6. `degenerate_poison_rate.png` — rate of no-op/degenerate poisons where poison equals truth.

Attack examples:
- `attack_examples.md`
- `attack_examples.json`

Headline metrics:
- Stage 1 mean reward: 1.415; retrieval@5: 0.782; scorer ASR: 0.418
- Stage 2 mean reward: 1.056; retrieval@5: 0.799; judge ASR: 0.015
- Final eval: valid=1.000; retrieval@5=0.750; scorer ASR=0.500; judge ASR=0.000
- Degenerate eval poisons: 30/60 (50%)
