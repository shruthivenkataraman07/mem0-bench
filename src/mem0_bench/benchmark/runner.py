"""The backend-agnostic benchmark runner.

Depends only on the :class:`~mem0_bench.memory.base.MemoryBackend`
abstraction; it has no branch of logic specific to Mem0, RAG, or
full-context. Handles:

- chronological, per-user ingestion (sessions are added in order; a memory
  write for session N is always awaited before session N+1 is issued),
- bounded concurrency *across* users (independent users' queries may run
  concurrently; one user's writes are never reordered),
- per-query tracing with full retrieval + answer-quality + latency + token
  instrumentation,
- explicit failure tracking (a failed ingest or query is traced as a
  failure, not silently dropped or counted as correct).
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from mem0_bench.benchmark.evaluator import evaluate_retrieval, score_answer
from mem0_bench.datasets.schema import BenchmarkDataset, GoldMemory, UserDataset
from mem0_bench.llm.client import LLMClient
from mem0_bench.memory.base import MemoryBackend
from mem0_bench.metrics.answer_quality import llm_judge
from mem0_bench.storage.traces import Trace, TraceWriter, get_git_commit


class BenchmarkRunner:
    def __init__(
        self,
        backend: MemoryBackend,
        run_id: str,
        seed: int,
        model: str,
        backend_config: Optional[Dict] = None,
        embedding_model: Optional[str] = None,
        top_k: int = 5,
        concurrency: int = 4,
        judge: Optional[LLMClient] = None,
    ) -> None:
        self.backend = backend
        self.run_id = run_id
        self.seed = seed
        self.model = model
        self.backend_config = backend_config or {}
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.concurrency = max(1, concurrency)
        self.judge = judge
        self._git_commit = get_git_commit()

    async def _ingest_user(self, user: UserDataset) -> Optional[str]:
        """Ingest sessions strictly in order. Returns an error string, if any."""
        for session_id in user.sessions:
            session_messages = [m for m in user.messages if m.session_id == session_id]
            result = await self.backend.add(user.user_id, session_messages)
            if not result.ok:
                return f"ingestion failed at session {session_id}: {result.error}"
        return None

    async def _run_question(
        self,
        user: UserDataset,
        question,
        gold_by_id: Dict[str, GoldMemory],
        ingest_error: Optional[str],
    ) -> Trace:
        base_kwargs = dict(
            run_id=self.run_id,
            git_commit=self._git_commit,
            backend=self.backend.name,
            backend_config=self.backend_config,
            model=self.model,
            embedding_model=self.embedding_model,
            seed=self.seed,
            user_id=user.user_id,
            query_id=question.question_id,
            query_type=question.category.value,
            query=question.query,
            expected_answer=question.expected_answer,
            expected_memory_ids=question.expected_memory_ids,
            retrieval_applicable=(self.backend.name != "full-context"),
        )

        if ingest_error is not None:
            return Trace(**base_kwargs, success=False, error=ingest_error)

        try:
            ans = await self.backend.answer(user.user_id, question.query, k=self.top_k)
        except Exception as e:  # noqa: BLE001 - traced, never silently dropped
            return Trace(**base_kwargs, success=False, error=str(e))

        if not ans.ok:
            return Trace(
                **base_kwargs, success=False, error=ans.error,
                retrieval_ms=ans.retrieval_ms, generation_ms=ans.generation_ms, total_ms=ans.total_ms,
            )

        retrieval_metrics = (
            evaluate_retrieval(ans.retrieved_memories, question, gold_by_id)
            if base_kwargs["retrieval_applicable"] else {}
        )
        quality = score_answer(ans.text, question)

        judge_score = None
        judge_model = None
        if self.judge is not None:
            verdict = await llm_judge(question.query, ans.text, question.expected_answer, self.judge)
            judge_score, judge_model = verdict.score, verdict.judge_model

        return Trace(
            **base_kwargs,
            prediction=ans.text,
            retrieved_memory_ids=[m.memory_id for m in ans.retrieved_memories],
            retrieval_metrics=retrieval_metrics,
            exact_match=quality.exact_match,
            normalized_match=quality.normalized_match,
            structured_match=quality.structured_match,
            is_correct=quality.is_correct,
            judge_score=judge_score,
            judge_model=judge_model,
            retrieval_ms=ans.retrieval_ms,
            prompt_construction_ms=ans.prompt_construction_ms,
            generation_ms=ans.generation_ms,
            total_ms=ans.total_ms,
            input_tokens=ans.input_tokens,
            retrieved_tokens=ans.retrieved_tokens,
            output_tokens=ans.output_tokens,
            history_total_turns=ans.history_total_turns,
            history_included_turns=ans.history_included_turns,
            history_dropped_turns=ans.history_dropped_turns,
            truncated=ans.truncated,
            success=True,
        )

    async def run_user(self, user: UserDataset, gold_by_id: Dict[str, GoldMemory]) -> List[Trace]:
        await self.backend.reset(user.user_id)
        ingest_error = await self._ingest_user(user)
        traces = []
        for question in user.questions:
            traces.append(await self._run_question(user, question, gold_by_id, ingest_error))
        return traces

    async def run(
        self, dataset: BenchmarkDataset, trace_writer: Optional[TraceWriter] = None
    ) -> List[Trace]:
        gold_by_id = {g.memory_id: g for u in dataset.users for g in u.gold_memories}
        semaphore = asyncio.Semaphore(self.concurrency)

        async def _bounded(user: UserDataset) -> List[Trace]:
            async with semaphore:
                return await self.run_user(user, gold_by_id)

        per_user_traces = await asyncio.gather(*[_bounded(u) for u in dataset.users])
        all_traces = [t for traces in per_user_traces for t in traces]
        if trace_writer is not None:
            for t in all_traces:
                trace_writer.write(t)
        return all_traces
