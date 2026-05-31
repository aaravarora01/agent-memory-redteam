"""Two-phase finance memory poisoning environment."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from .finance_tools import FinanceTools
from .memory_store import FinanceMemoryStore
from .schemas import (
    DATA_DIR,
    AgentAnswer,
    EnvMode,
    EpisodeResult,
    FinanceQuery,
    PoisonAction,
    UserProfile,
    load_queries,
    load_user,
)
from .scorer import (
    check_collateral_retrieval,
    resolve_answer_from_hits,
    score_answer,
    score_retrieval,
)
from ..rl.action_space import decode_poison_action


RewardMode = Literal["sparse", "shaped"]


@dataclass
class HeuristicFinanceAgent:
    """Deterministic heuristic agent (no LLM)."""

    p_tool: float = 0.8

    def answer(
        self,
        query: FinanceQuery,
        retrieved: list,
        tools: FinanceTools,
        mode: str,
        rng: random.Random,
    ) -> AgentAnswer:
        tool_calls: list[str] = []
        tool_value = tools.resolve_fact(query.target_fact_id)

        use_tool = False
        if mode == EnvMode.TOOL_FORCED.value:
            if query.query_family in ("amount", "merchant", "account"):
                use_tool = True
        elif mode == EnvMode.TOOL_OPTIONAL.value:
            if query.query_family in ("amount", "merchant", "account"):
                use_tool = rng.random() < self.p_tool
            else:
                use_tool = False

        if use_tool:
            tool_calls.append(f"resolve_fact({query.target_fact_id})")
            if mode == EnvMode.TOOL_FORCED.value:
                return AgentAnswer(
                    final_answer_value=tool_value,
                    final_answer_text=str(tool_value),
                    tool_calls=tool_calls,
                    tool_value=tool_value,
                    used_tool=True,
                    source="tool",
                )
            # tool_optional: may still prefer poisoned memory if rank 1 is poison
            top = retrieved[0] if retrieved else None
            if top and top.is_poison and top.fact_id == query.target_fact_id:
                return AgentAnswer(
                    final_answer_value=top.value,
                    final_answer_text=str(top.value),
                    tool_calls=tool_calls,
                    tool_value=tool_value,
                    used_tool=True,
                    source="memory",
                )
            return AgentAnswer(
                final_answer_value=tool_value,
                final_answer_text=str(tool_value),
                tool_calls=tool_calls,
                tool_value=tool_value,
                used_tool=True,
                source="tool",
            )

        mem_value = resolve_answer_from_hits(retrieved, query.target_fact_id)
        return AgentAnswer(
            final_answer_value=mem_value,
            final_answer_text=str(mem_value),
            tool_calls=tool_calls,
            tool_value=tool_value if tool_calls else None,
            used_tool=bool(tool_calls),
            source="memory",
        )


class FinanceMemoryPoisonEnv:
    def __init__(
        self,
        *,
        mode: str = EnvMode.RETRIEVAL_ONLY.value,
        reward_mode: RewardMode = "sparse",
        k: int = 5,
        user: Optional[UserProfile] = None,
        queries: Optional[list[FinanceQuery]] = None,
        tools: Optional[FinanceTools] = None,
        memory_path: Path | str = DATA_DIR / "clean_memories.jsonl",
        backend: Optional[str] = None,
        seed: int = 0,
    ):
        self.mode = mode
        self.reward_mode = reward_mode
        self.k = k
        self.user = user or load_user()
        self.queries = queries or load_queries()
        self.tools = tools or FinanceTools.from_data()
        self.memory_path = Path(memory_path)
        self.backend = backend
        self.rng = random.Random(seed)
        self.agent = HeuristicFinanceAgent()
        self._episode = 0
        self._target_fact_id: Optional[str] = None
        self._query: Optional[FinanceQuery] = None
        self._store: Optional[FinanceMemoryStore] = None

    def _sample_query(
        self,
        target_fact_id: Optional[str] = None,
        query_id: Optional[str] = None,
    ) -> FinanceQuery:
        if query_id:
            for q in self.queries:
                if q.query_id == query_id:
                    return q
            raise ValueError(f"Unknown query_id: {query_id}")
        pool = self.queries
        if target_fact_id:
            pool = [q for q in self.queries if q.target_fact_id == target_fact_id]
        if not pool:
            raise ValueError(f"No queries for target_fact_id={target_fact_id}")
        return self.rng.choice(pool)

    def reset(
        self,
        *,
        target_fact_id: Optional[str] = None,
        query_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> dict:
        if seed is not None:
            self.rng = random.Random(seed)
        self._store = FinanceMemoryStore.from_clean_corpus(
            self.memory_path, backend=self.backend
        )
        self._query = self._sample_query(target_fact_id, query_id)
        self._target_fact_id = self._query.target_fact_id
        return {
            "target_fact_id": self._target_fact_id,
            "query_id": self._query.query_id,
        }

    def step(
        self,
        action: PoisonAction | dict,
        *,
        reward_mode: Optional[RewardMode] = None,
    ) -> tuple[dict, float, bool, dict]:
        if self._store is None or self._query is None:
            raise RuntimeError("Call reset() before step()")

        if isinstance(action, dict):
            action = PoisonAction(**action)

        poison = decode_poison_action(
            action,
            self.user,
            episode=self._episode,
            memory_id=f"poison_{self._episode:04d}",
        )
        store = self._store.clone_with_poison(poison)
        hits = store.retrieve(self._query.query_text, k=self.k)

        collateral = False
        other_facts = {f for f in {q.target_fact_id for q in self.queries} if f != self._query.target_fact_id}
        if action.target_fact in other_facts:
            collateral = check_collateral_retrieval(
                hits, action.target_fact, self._query.target_fact_id
            )

        retrieval = score_retrieval(
            hits,
            self._query.target_fact_id,
            k=self.k,
            collateral=collateral,
        )

        rm = reward_mode or self.reward_mode
        reward = retrieval.sparse_reward if rm == "sparse" else retrieval.shaped_reward

        true_value = self.tools.resolve_fact(self._query.target_fact_id)
        tool_calls: list[str] = []
        tool_value = None
        final_answer_value = None
        answer_uses_poison = None
        answer_contradicts_tool = None

        if self.mode != EnvMode.RETRIEVAL_ONLY.value:
            answer = self.agent.answer(
                self._query, hits, self.tools, self.mode, self.rng
            )
            tool_calls = answer.tool_calls
            tool_value = answer.tool_value
            final_answer_value = answer.final_answer_value
            ans_score = score_answer(
                answer, poison.poison_value, true_value, tool_value
            )
            answer_uses_poison = ans_score["answer_uses_poison"]
            answer_contradicts_tool = ans_score["answer_contradicts_tool"]

        obs = {
            "target_fact_id": self._target_fact_id,
            "poison_style": action.framing_style,
            "corrupted_value": poison.poison_value,
            "poison_in_top_k": retrieval.poison_in_top_k,
            "poison_rank": retrieval.poison_rank,
            "truth_rank": retrieval.truth_rank,
            "poison_score": retrieval.poison_score,
            "truth_score": retrieval.truth_score,
            "poison_ranks_above_truth": retrieval.poison_ranks_above_truth,
            "truth_displaced": retrieval.truth_displaced,
        }

        info = {
            "query": self._query.query_text,
            "retrieved_memory_ids": [h.memory_id for h in hits],
            "retrieval_score": retrieval,
            "poison_value": poison.poison_value,
            "true_value": true_value,
            "incorrect_retrieval_dominant": retrieval.incorrect_retrieval_dominant,
            "collateral_retrieval": collateral,
            "tool_calls": tool_calls,
            "tool_value": tool_value,
            "final_answer_value": final_answer_value,
            "answer_uses_poison": answer_uses_poison,
            "answer_contradicts_tool": answer_contradicts_tool,
        }

        self._episode += 1
        return obs, reward, True, info

    def run_episode_log(
        self,
        action: PoisonAction,
        *,
        reward_mode: Optional[RewardMode] = None,
    ) -> EpisodeResult:
        if self._query is None:
            self.reset()
        obs, reward, _, info = self.step(action, reward_mode=reward_mode)
        retrieval = info["retrieval_score"]
        return EpisodeResult(
            episode=self._episode - 1,
            mode=self.mode,
            user_id=self.user.user_id,
            target_fact=self._query.target_fact_id,
            true_value=info["true_value"],
            poison_value=info["poison_value"],
            poison_style=action.framing_style,
            query=info["query"],
            retrieved_memory_ids=info["retrieved_memory_ids"],
            poison_rank=retrieval.poison_rank,
            truth_rank=retrieval.truth_rank,
            poison_in_top5=retrieval.poison_in_top_k,
            truth_in_top5=not retrieval.truth_displaced,
            poison_ranks_above_truth=retrieval.poison_ranks_above_truth,
            truth_displaced=retrieval.truth_displaced,
            incorrect_retrieval_dominant=retrieval.incorrect_retrieval_dominant,
            retrieval_reward=reward,
            collateral_retrieval=info["collateral_retrieval"],
            tool_calls=info["tool_calls"],
            tool_value=info["tool_value"],
            final_answer_value=info["final_answer_value"],
            answer_uses_poison=info["answer_uses_poison"],
            answer_contradicts_tool=info["answer_contradicts_tool"],
        )
