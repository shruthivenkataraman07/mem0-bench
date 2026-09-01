"""Streamlit dashboard for mem0-bench results.

Reads result directories under `results/<run_id>/`; it never uses
hardcoded numbers. Run with:

    mem0-bench dashboard
    # or directly:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

st.set_page_config(page_title="mem0-bench dashboard", layout="wide")
st.title("mem0-bench: memory architecture comparison")
st.caption(
    "Mem0 vs. conventional RAG vs. full-context history. All figures below are read "
    "directly from result files under `results/<run_id>/`; nothing here is hardcoded."
)


def is_attack_summary(summary: Dict[str, Any]) -> bool:
    """Attack-run summaries (mem0-bench attack) have a different shape than
    backend-comparison summaries (run/compare/scale); distinguish them so
    we never index one as if it were the other."""
    return bool(summary) and all(
        isinstance(v, dict) and "poison_retrieval_rate" in v for v in summary.values()
    )


@st.cache_data
def list_runs(results_dir: str) -> list:
    base = Path(results_dir)
    if not base.exists():
        return []
    return sorted((p.name for p in base.iterdir() if p.is_dir()), reverse=True)


@st.cache_data
def load_run(results_dir: str, run_id: str):
    run_path = Path(results_dir) / run_id
    summary = json.loads((run_path / "summary.json").read_text()) if (run_path / "summary.json").exists() else {}
    metadata = json.loads((run_path / "metadata.json").read_text()) if (run_path / "metadata.json").exists() else {}
    traces_path = run_path / "traces.jsonl"
    traces = []
    if traces_path.exists():
        with traces_path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    traces.append(json.loads(line))
    return summary, metadata, pd.DataFrame(traces)


results_dir = st.sidebar.text_input("Results directory", str(DEFAULT_RESULTS_DIR))
runs = list_runs(results_dir)

if not runs:
    st.warning(
        f"No runs found under `{results_dir}`. Generate a dataset and run a benchmark first, e.g.:\n\n"
        "```\nmem0-bench generate --users 5 --turns 50 --seed 42\n"
        "mem0-bench compare --backends mem0,rag,full-context --dataset datasets/conversations/<file>.json\n```"
    )
    st.stop()

selected_runs = st.sidebar.multiselect("Runs", runs, default=runs[:1])
if not selected_runs:
    st.info("Select at least one run from the sidebar.")
    st.stop()

# Load every selected run's summary + traces up front, then split by shape:
# backend-comparison summaries (run/compare/scale) vs. attack summaries
# (mem0-bench attack). A run can be usefully shown even with zero traces
# (e.g. an attack-only run only writes summary.json), so nothing below
# gates the whole page on `df` being non-empty; each section checks its
# own inputs.
all_traces = []
backend_summaries: Dict[str, Dict[str, Any]] = {}
attack_summaries: Dict[str, Dict[str, Any]] = {}
for run_id in selected_runs:
    summary, metadata, traces_df = load_run(results_dir, run_id)
    if is_attack_summary(summary):
        attack_summaries[run_id] = summary
    elif summary:
        backend_summaries[run_id] = summary
    if not traces_df.empty:
        traces_df["run_id_selected"] = run_id
        all_traces.append(traces_df)

df = pd.concat(all_traces, ignore_index=True) if all_traces else pd.DataFrame()

if not df.empty:
    backend_options = sorted(df["backend"].unique())
    qtype_options = sorted(df["query_type"].unique())
    backend_filter = st.sidebar.multiselect("Backend", backend_options, default=backend_options)
    qtype_filter = st.sidebar.multiselect("Query type", qtype_options, default=qtype_options)
    df = df[df["backend"].isin(backend_filter) & df["query_type"].isin(qtype_filter)]
else:
    backend_filter = None  # no trace-derived filter available; backend summaries are shown unfiltered

st.header("Backend comparison")
if not backend_summaries:
    st.caption("No `run` / `compare` / `scale` results among the selected runs yet.")
else:
    summary_rows = []
    for run_id, summary in backend_summaries.items():
        for backend, stats in summary.items():
            if backend_filter is not None and backend not in backend_filter:
                continue
            retrieval = stats.get("retrieval") or {}
            summary_rows.append({
                "run_id": run_id,
                "backend": backend,
                "accuracy": stats["accuracy"]["accuracy"],
                "failure_rate": stats["accuracy"]["failure_rate"],
                "recall@5": retrieval.get("k5", {}).get("recall") if retrieval else None,
                "mrr@5": retrieval.get("k5", {}).get("mrr") if retrieval else None,
                "ndcg@5": retrieval.get("k5", {}).get("ndcg") if retrieval else None,
                "p50_ms": stats["latency"]["total_ms"]["p50_ms"],
                "p95_ms": stats["latency"]["total_ms"]["p95_ms"],
                "avg_tokens": stats["tokens"]["per_query_average"]["total_tokens"],
                "estimated_cost_usd": stats["tokens"].get("estimated_cost_usd"),
            })
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)

    if not summary_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(summary_df, x="backend", y="accuracy", color="run_id", barmode="group",
                                    title="Answer accuracy by backend"), use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(summary_df, x="backend", y="recall@5", color="run_id", barmode="group",
                                    title="Recall@5 by backend (N/A for full-context)"), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(px.bar(summary_df, x="backend", y="p95_ms", color="run_id", barmode="group",
                                    title="p95 end-to-end latency (ms)"), use_container_width=True)
        with c4:
            st.plotly_chart(px.bar(summary_df, x="backend", y="avg_tokens", color="run_id", barmode="group",
                                    title="Average total tokens per query"), use_container_width=True)

    st.header("Query-type breakdown")
    qtype_rows = []
    for run_id, summary in backend_summaries.items():
        for backend, stats in summary.items():
            if backend_filter is not None and backend not in backend_filter:
                continue
            for qtype, qstats in stats.get("by_query_type", {}).items():
                qtype_rows.append({
                    "run_id": run_id, "backend": backend, "query_type": qtype,
                    "accuracy": qstats["accuracy"]["accuracy"],
                })
    qtype_df = pd.DataFrame(qtype_rows)
    if not qtype_df.empty:
        st.plotly_chart(
            px.bar(qtype_df, x="query_type", y="accuracy", color="backend", barmode="group",
                   facet_col="run_id", title="Accuracy by query type"),
            use_container_width=True,
        )

st.header("Scale curves (if this run swept history length)")
if not df.empty and "history_total_turns" in df.columns and df["history_total_turns"].notna().any():
    scale_df = df.groupby(["backend", "history_total_turns"], as_index=False).agg(
        accuracy=("is_correct", "mean"), p95_ms=("total_ms", lambda s: s.quantile(0.95)),
        avg_tokens=("input_tokens", "mean"),
    )
    st.plotly_chart(px.line(scale_df, x="history_total_turns", y="accuracy", color="backend", markers=True,
                             title="Accuracy vs. history length"), use_container_width=True)
    st.plotly_chart(px.line(scale_df, x="history_total_turns", y="p95_ms", color="backend", markers=True,
                             title="p95 latency vs. history length"), use_container_width=True)
else:
    st.caption("Run `mem0-bench scale ...` to populate this section.")

st.header("Red-team results (if this run included an attack)")
if not attack_summaries:
    st.caption("Run `mem0-bench attack ...` to populate this section.")
else:
    for run_id, raw in attack_summaries.items():
        st.subheader(f"Run: {run_id}")
        attack_df = pd.DataFrame([{"attack": k, **v} for k, v in raw.items()])
        st.dataframe(attack_df, use_container_width=True)
        st.plotly_chart(
            px.bar(attack_df, x="attack", y=["poison_retrieval_rate", "attack_success_rate"], barmode="group",
                   title=f"Red-team results: {run_id}"),
            use_container_width=True,
        )

st.header("Raw per-query traces")
if df.empty:
    st.caption("No per-query traces among the selected runs (attack runs only write a summary).")
else:
    st.dataframe(df, use_container_width=True)
