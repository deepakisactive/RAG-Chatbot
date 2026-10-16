"""Multi-Model Evaluation & RAG Pipeline Diagnostics Dashboard.

Exercises 1-6:
- Ex 1/3/4: Multi-Model Quantitative Benchmark (Llama 3.2 1B, StarCoder2 3B, Qwen2.5-Coder 1.5B)
- Ex 2: Evaluation Dataset Explorer (25 representative tasks)
- Ex 5: End-to-End RAG Pipeline Trace Analysis
- Ex 6: Repository & Codebase Understanding Investigation
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import streamlit as st

from app.config import settings
from app.llm.ollama_client import get_ollama_client
from app.streamlit.components.status import render_sidebar_status

RESULTS_FILE = _project_root / "evaluation" / "results" / "evaluation_results.json"
TRACE_FILE = _project_root / "evaluation" / "results" / "rag_pipeline_trace_cases.json"
REPO_FILE = _project_root / "evaluation" / "results" / "repo_understanding_results.json"
GUARDRAILS_RESULTS_FILE = _project_root / "evaluation" / "results" / "guardrail_test_results.json"
GUARDRAILS_DATASET_FILE = _project_root / "evaluation" / "guardrails_dataset.json"
DATASET_FILE = _project_root / "evaluation" / "dataset.json"
BENCHMARK_LOG = _project_root / "evaluation" / "results" / "benchmark_run.log"

st.set_page_config(page_title="Model Evaluation", page_icon="🧪", layout="wide")

render_sidebar_status()

st.title("🧪 Multi-Model RAG & Code Evaluation Workbench")
st.markdown(
    "Comprehensive benchmark suite comparing **Llama 3.2 1B**, **StarCoder2 3B**, and **Qwen2.5-Coder 1.5B**, "
    "end-to-end RAG traces, repository understanding, and **AI Output & Guardrail Verification**."
)

# ── Status Banner ──────────────────────────────────────────────────────────────
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
col_s1.metric("📊 Benchmark", "✅ Ready" if RESULTS_FILE.exists() else "⏳ Not Generated")
col_s2.metric("🔍 Traces", "✅ Ready" if TRACE_FILE.exists() else "⏳ Not Generated")
col_s3.metric("🏗️ Repo Intel", "✅ Ready" if REPO_FILE.exists() else "⏳ Not Generated")
col_s4.metric("🛡️ Guardrails", "✅ Ready" if GUARDRAILS_RESULTS_FILE.exists() else "⏳ Not Generated")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Benchmark (Ex 1, 3, 4)",
    "📑 Dataset (Ex 2)",
    "🔍 RAG Traces (Ex 5)",
    "🏗️ Repo Understanding (Ex 6)",
    "⚡ Quick LLM Test",
    "🛡️ Guardrails & AI Output",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Quantitative Benchmark
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🏆 Multi-Model Quantitative Comparison (Exercises 1, 3, 4)")

    # ── Exercise descriptions ──────────────────────────────────────────────────
    with st.expander("ℹ️ About These Exercises", expanded=False):
        st.markdown("""
        **Exercise 1 – Evaluate Multiple LLM Models**
        Evaluates the same RAG application with Llama 3.2 1B, StarCoder2 3B, and Qwen2.5-Coder 1.5B.
        Identical prompts, questions, knowledge base (241 Qdrant chunks), Top-K=5, threshold=0.45.

        **Exercise 3 – Quantitative Evaluation Metrics**
        - **Correctness / Accuracy:** Token-level F1 score between prediction and ground truth
          `F1 = 2 × (Precision × Recall) / (Precision + Recall)` on tokenized word sets
        - **Relevance:** Jaccard similarity across (query ∪ ground_truth) and prediction token sets
        - **Retrieval Quality:** Recall@K — fraction of tasks where expected document was retrieved
        - **Hallucination Rate:** Answers to unanswerable queries without refusing = hallucination
        - **Code Test-Pass Rate:** Generated Python functions executed against embedded unit test suites
        - **Response Latency:** Mean, Median, P95 in milliseconds measured per-task
        - **Token Usage:** eval_count (output tokens), prompt_eval_count (input tokens), tokens/sec
        - **CPU/Memory:** `resource.getrusage()` CPU user+sys time delta and peak RSS in MB

        **Exercise 4 – Analysis**
        The comparison matrix and charts below provide quantitative evidence for trade-off analysis.
        """)

    # ── Run Benchmark Button ───────────────────────────────────────────────────
    ollama_client_check = get_ollama_client()
    available_models = [m.get("name") for m in ollama_client_check.list_models()]
    required_models = ["llama3.2:1b", "starcoder2:3b", "qwen2.5-coder:1.5b"]
    missing = [m for m in required_models if m not in available_models]

    if missing:
        st.warning(f"⚠️ Missing models: `{'`, `'.join(missing)}`. Pull them with `ollama pull <model>` on the host.")
    else:
        st.success(f"✅ All 3 required models available: `{'` · `'.join(required_models)}`")

    run_col, info_col = st.columns([1, 3])
    with run_col:
        run_benchmark = st.button(
            "🚀 Run Full Benchmark",
            type="primary",
            disabled=bool(missing),
            help="Runs 25 tasks × 3 models. Takes ~20-40 min on CPU. Results auto-reload when done.",
        )
    with info_col:
        if RESULTS_FILE.exists():
            mtime = RESULTS_FILE.stat().st_mtime
            st.caption(f"Last benchmark: `{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(mtime))}` — click button to re-run.")
        else:
            st.caption("No results yet. Click **🚀 Run Full Benchmark** to start (runs ~25–40 min in background).")

    if run_benchmark:
        st.info("🔄 Starting benchmark runner in the background... Results will appear automatically when complete.")
        log_placeholder = st.empty()

        def _run_benchmark():
            log_path = BENCHMARK_LOG
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as log_f:
                proc = subprocess.run(
                    [sys.executable, str(_project_root / "evaluation" / "runner.py")],
                    cwd=str(_project_root),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            return proc.returncode

        t = threading.Thread(target=_run_benchmark, daemon=True)
        t.start()

        # Stream log tail while running
        start = time.time()
        while t.is_alive() and (time.time() - start) < 10:
            if BENCHMARK_LOG.exists():
                lines = BENCHMARK_LOG.read_text(encoding="utf-8").splitlines()
                log_placeholder.code("\n".join(lines[-15:]), language="text")
            time.sleep(1.5)

        if t.is_alive():
            st.success("✅ Benchmark running in background. Refresh this page in ~30 minutes to see results.")
        else:
            st.success("✅ Benchmark completed! Reload the page to view results.")
            st.rerun()

    st.markdown("---")

    if not RESULTS_FILE.exists():
        st.warning(
            "No benchmark results yet. The full benchmark runs 25 tasks × 3 models "
            "(≈25–40 min on CPU). Click **🚀 Run Full Benchmark** above to start."
        )
        if BENCHMARK_LOG.exists():
            st.markdown("#### 📋 Live Progress Log (last 20 lines)")
            lines = BENCHMARK_LOG.read_text(encoding="utf-8").splitlines()
            st.code("\n".join(lines[-20:]), language="text")
            if st.button("🔄 Refresh Log"):
                st.rerun()
    else:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        summaries = data.get("model_summaries", {})
        models = list(summaries.keys())
        meta = data.get("meta", {})

        # Meta info
        st.caption(
            f"Generated: `{meta.get('timestamp', 'unknown')}` | "
            f"Tasks: `{meta.get('total_tasks', '?')}` | "
            f"Top-K: `{meta.get('top_k', '?')}` | "
            f"Threshold: `{meta.get('similarity_threshold', '?')}`"
        )

        # ── Key Metric Cards ──
        st.markdown("#### 📈 Key Quality & Performance Metrics")
        cols = st.columns(len(models))
        model_icons = {"llama3.2:1b": "🦙", "starcoder2:3b": "⭐", "qwen2.5-coder:1.5b": "🧠"}

        for idx, m in enumerate(models):
            s = summaries[m]
            icon = model_icons.get(m, "🤖")
            with cols[idx]:
                st.subheader(f"{icon} `{m}`")
                st.metric("Accuracy (Token F1)", f"{s.get('mean_token_f1_accuracy', 0):.4f}")
                st.metric("Keyword Recall", f"{s.get('mean_keyword_recall', 0):.4f}")
                st.metric("Relevance Score", f"{s.get('mean_relevance', 0):.4f}")
                st.metric("Code Test-Pass Rate", f"{s.get('code_test_pass_rate_pct', 0):.1f}%")
                st.metric("Hallucination Rate", f"{s.get('hallucination_rate_pct', 0):.1f}%")
                st.metric("Mean Latency", f"{s.get('mean_generation_latency_ms', 0):.0f} ms")
                st.metric("P95 Latency", f"{s.get('p95_generation_latency_ms', 0):.0f} ms")
                st.metric("Peak Memory", f"{s.get('peak_memory_rss_mb', 0):.1f} MB")

        st.markdown("---")

        # ── Summary Table ──
        st.markdown("#### 📋 Comprehensive Comparison Matrix (Exercise 4)")
        rows = []
        for m in models:
            s = summaries[m]
            rows.append({
                "Model": m,
                "Accuracy (F1)": round(s.get("mean_token_f1_accuracy", 0), 4),
                "Keyword Recall": round(s.get("mean_keyword_recall", 0), 4),
                "Relevance": round(s.get("mean_relevance", 0), 4),
                "Hallucination (%)": s.get("hallucination_rate_pct", 0),
                "Code Pass (%)": s.get("code_test_pass_rate_pct", 0),
                "Mean Latency (ms)": s.get("mean_generation_latency_ms", 0),
                "Median Latency (ms)": s.get("median_generation_latency_ms", 0),
                "P95 Latency (ms)": s.get("p95_generation_latency_ms", 0),
                "CPU Time (s)": s.get("cpu_time_sec_delta", 0),
                "Peak RAM (MB)": s.get("peak_memory_rss_mb", 0),
            })
        df_summary = pd.DataFrame(rows)
        st.dataframe(df_summary, use_container_width=True)

        # ── Analysis conclusions ───────────────────────────────────────────────
        st.markdown("#### 🔬 Quantitative Analysis (Exercise 4)")
        best_f1 = max(models, key=lambda m: summaries[m].get("mean_token_f1_accuracy", 0))
        best_recall = max(models, key=lambda m: summaries[m].get("mean_keyword_recall", 0))
        lowest_halluc = min(models, key=lambda m: summaries[m].get("hallucination_rate_pct", 100))
        fastest = min(models, key=lambda m: summaries[m].get("mean_generation_latency_ms", 9e9))
        best_code = max(models, key=lambda m: summaries[m].get("code_test_pass_rate_pct", 0))
        lowest_mem = min(models, key=lambda m: summaries[m].get("peak_memory_rss_mb", 9e9))

        analysis_data = [
            {"Metric": "Best Accuracy (Token F1)", "Winner": best_f1,
             "Value": f"{summaries[best_f1].get('mean_token_f1_accuracy', 0):.4f}"},
            {"Metric": "Best Keyword Recall", "Winner": best_recall,
             "Value": f"{summaries[best_recall].get('mean_keyword_recall', 0):.4f}"},
            {"Metric": "Lowest Hallucination Rate", "Winner": lowest_halluc,
             "Value": f"{summaries[lowest_halluc].get('hallucination_rate_pct', 0):.1f}%"},
            {"Metric": "Best Code Test-Pass Rate", "Winner": best_code,
             "Value": f"{summaries[best_code].get('code_test_pass_rate_pct', 0):.1f}%"},
            {"Metric": "Fastest Responses (Mean)", "Winner": fastest,
             "Value": f"{summaries[fastest].get('mean_generation_latency_ms', 0):.0f} ms"},
            {"Metric": "Lowest Peak Memory", "Winner": lowest_mem,
             "Value": f"{summaries[lowest_mem].get('peak_memory_rss_mb', 0):.1f} MB"},
        ]
        st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)

        # Quality-Latency tradeoff analysis
        st.markdown("**Quality–Latency–Resource Trade-off Analysis**")
        tradeoff_text = []
        for m in models:
            s = summaries[m]
            icon = model_icons.get(m, "🤖")
            f1 = s.get("mean_token_f1_accuracy", 0)
            lat = s.get("mean_generation_latency_ms", 0)
            hall = s.get("hallucination_rate_pct", 0)
            mem = s.get("peak_memory_rss_mb", 0)
            tradeoff_text.append(
                f"**{icon} {m}**: F1={f1:.4f}, Latency={lat:.0f}ms, "
                f"Hallucination={hall:.1f}%, Peak RAM={mem:.1f}MB"
            )
        for line in tradeoff_text:
            st.markdown(f"- {line}")

        st.info(
            "💡 **Key insight**: The most accurate model is not necessarily the most efficient. "
            f"`{best_f1}` leads in accuracy but `{fastest}` has the lowest latency. "
            "This quantitative evidence demonstrates the inherent quality–latency–resource trade-off in LLM selection."
        )

        # ── Charts ──
        st.markdown("#### 📊 Visual Comparisons")
        ch1, ch2, ch3 = st.columns(3)

        with ch1:
            st.markdown("**Quality Metrics**")
            qual_df = pd.DataFrame({
                "Token F1": {m: summaries[m].get("mean_token_f1_accuracy", 0) for m in models},
                "Keyword Recall": {m: summaries[m].get("mean_keyword_recall", 0) for m in models},
                "Relevance": {m: summaries[m].get("mean_relevance", 0) for m in models},
            })
            st.bar_chart(qual_df)

        with ch2:
            st.markdown("**Latency (seconds)**")
            lat_df = pd.DataFrame({
                "Mean Gen (s)": {m: summaries[m].get("mean_generation_latency_ms", 0) / 1000 for m in models},
                "P95 Gen (s)": {m: summaries[m].get("p95_generation_latency_ms", 0) / 1000 for m in models},
            })
            st.bar_chart(lat_df)

        with ch3:
            st.markdown("**Code Pass & Hallucination (%)**")
            risk_df = pd.DataFrame({
                "Code Pass (%)": {m: summaries[m].get("code_test_pass_rate_pct", 0) for m in models},
                "Hallucination (%)": {m: summaries[m].get("hallucination_rate_pct", 0) for m in models},
            })
            st.bar_chart(risk_df)

        st.markdown("---")

        # ── Task-by-Task ──
        st.markdown("#### 🔬 Task-by-Task Model Outputs")
        tasks = data.get("detailed_task_results", [])
        categories = sorted(set(t["category"] for t in tasks))
        sel_cat = st.selectbox("Filter by Category", options=["ALL"] + categories)

        filtered = tasks if sel_cat == "ALL" else [t for t in tasks if t["category"] == sel_cat]

        for t in filtered:
            label = f"**[{t['id']}] {t['category']}** — {t['question'][:75]}..."
            with st.expander(label):
                st.markdown(f"**Question:** {t['question']}")
                st.markdown(f"**Ground Truth:** {t['ground_truth']}")
                st.markdown(f"**Unanswerable:** {'Yes ⚠️' if t.get('is_unanswerable') else 'No'}")

                task_cols = st.columns(len(models))
                for idx, m in enumerate(models):
                    m_out = t["model_outputs"].get(m, {})
                    with task_cols[idx]:
                        icon = model_icons.get(m, "🤖")
                        st.markdown(f"**{icon} {m}**")
                        st.markdown(
                            f"*F1:* `{m_out.get('token_f1', 0):.4f}` | "
                            f"*Recall:* `{m_out.get('keyword_recall', 0):.4f}` | "
                            f"*Latency:* `{m_out.get('generation_time_ms', 0):.0f}` ms"
                        )
                        if m_out.get("test_passed") is not None:
                            badge = "✅ Passed" if m_out["test_passed"] else "❌ Failed"
                            st.caption(f"Code Test: {badge}")
                        if m_out.get("is_hallucination"):
                            st.warning("⚠️ Hallucination Flagged")
                        st.info(m_out.get("answer", "No answer")[:400])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Evaluation Dataset
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📑 Evaluation Dataset (Exercise 2)")
    st.markdown(
        "25 representative tasks spanning the real use-case spectrum of this RAG application: "
        "technical paper QA, codebase Q&A, code generation with unit tests, and hallucination probes."
    )

    with st.expander("ℹ️ Exercise 2 — Dataset Design Rationale", expanded=False):
        st.markdown("""
        The 25 tasks cover 7 distinct categories representing the real tasks a user would expect:
        - **RAG_DOMAIN_KNOWLEDGE** (5): Factual extraction from indexed PDFs (ProteinMPNN, disease history, sepsis, cloudburst)
        - **CODE_EXPLANATION** (5): Explain functions/modules in this codebase
        - **CODE_RETRIEVAL** (4): Locate files/components handling specific functionality
        - **DEPENDENCY_UNDERSTANDING** (3): Trace inter-module relationships
        - **BUG_ANALYSIS** (3): Identify root causes of potential failures
        - **CODE_GENERATION** (3): Write Python functions with embedded unit tests
        - **RAG_BASED_QUESTION** (2): Questions requiring document grounding + hallucination probes (unanswerable)

        The **same 25 questions** are used for all 3 models ensuring a fair comparison (Exercise 1 requirement).
        """)

    if DATASET_FILE.exists():
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            dataset_items = json.load(f)

        # Category distribution
        cat_counts = {}
        for item in dataset_items:
            cat = item["category"]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Tasks", len(dataset_items))
        c2.metric("Categories", len(cat_counts))
        code_tasks = sum(1 for d in dataset_items if d.get("is_code_task"))
        c3.metric("Code Generation Tasks", code_tasks)

        # Category bar chart
        cat_df = pd.DataFrame({"Category": list(cat_counts.keys()), "Tasks": list(cat_counts.values())}).set_index("Category")
        st.bar_chart(cat_df)

        # Full table
        df_tasks = pd.DataFrame([
            {
                "ID": item["id"],
                "Category": item["category"],
                "Task Type": item["task_type"],
                "Question (truncated)": item["question"][:80] + "...",
                "Code Task": "✅" if item.get("is_code_task") else "",
                "Unanswerable": "⚠️" if item.get("is_unanswerable") else "",
            }
            for item in dataset_items
        ])
        st.dataframe(df_tasks, use_container_width=True)

        # Detail expanders
        st.markdown("#### Task Details")
        for item in dataset_items:
            with st.expander(f"[{item['id']}] {item['question'][:60]}..."):
                st.markdown(f"**Category:** {item['category']} | **Type:** {item['task_type']}")
                st.markdown(f"**Question:** {item['question']}")
                st.markdown(f"**Ground Truth:** {item['ground_truth']}")
                if item.get("expected_keywords"):
                    st.markdown(f"**Keywords:** `{'`, `'.join(item['expected_keywords'])}`")
                if item.get("is_code_task"):
                    st.caption(f"Code task — test function: `{item.get('test_function_name', 'N/A')}`")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: RAG Pipeline Traces (Exercise 5)
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🔍 Exercise 5: RAG Pipeline Trace Analysis")
    st.markdown(
        """
        Systematic investigation of the causal chain:
        **Retrieval Quality → Context Quality → LLM Response Quality**

        Five diagnostic case studies covering: correct retrieval, irrelevant retrieval,
        missed information, correct answer production, and hallucination despite context.
        """
    )

    with st.expander("ℹ️ Exercise 5 — Methodology", expanded=False):
        st.markdown("""
        For each of 5 selected questions, we record:
        - **QUESTION** → the diagnostic query
        - **RETRIEVED CONTEXT** → top-K chunks from Qdrant (similarity scores, document source, text excerpts)
        - **LLM RESPONSE** → the model's generated answer

        **Causal Relationship Analysed:**
        ```
        RETRIEVAL QUALITY (Recall@K, similarity, noise)
              ↓
        CONTEXT QUALITY (relevance density, completeness, noise ratio)
              ↓
        LLM RESPONSE QUALITY (factual grounding, hallucination rate)
        ```

        This demonstrates that RAG is NOT simply a checkbox — poor retrieval propagates directly to poor responses.
        """)

    run_trace_col, _ = st.columns([1, 3])
    with run_trace_col:
        run_trace = st.button(
            "🔍 Run RAG Trace Analysis",
            help="Executes 5 diagnostic queries through the full RAG pipeline. Takes ~2-3 min.",
        )

    if run_trace:
        with st.spinner("Running 5 diagnostic trace queries through the RAG pipeline..."):
            result = subprocess.run(
                [sys.executable, str(_project_root / "evaluation" / "trace_analyzer.py")],
                cwd=str(_project_root),
                capture_output=True,
                text=True,
                timeout=600,
            )
        if result.returncode == 0:
            st.success("✅ RAG trace analysis complete!")
            st.rerun()
        else:
            st.error(f"❌ Trace analysis failed:\n```\n{result.stderr[-1000:]}\n```")

    if not TRACE_FILE.exists():
        st.warning("Trace data not yet generated. Click **🔍 Run RAG Trace Analysis** above.")
    else:
        with open(TRACE_FILE, "r", encoding="utf-8") as f:
            trace_data = json.load(f)

        meta_cfg = trace_data.get("analysis_meta", {}).get("pipeline_config", {})
        st.caption(
            f"Pipeline Config — Top-K: `{meta_cfg.get('top_k', '?')}` | "
            f"Threshold: `{meta_cfg.get('threshold', '?')}` | "
            f"Embedder: `{meta_cfg.get('embedding_model', '?')}`"
        )

        case_icons = {
            "CASE_1_RELEVANT_RETRIEVED": "✅",
            "CASE_2_IRRELEVANT_RETRIEVED": "🔀",
            "CASE_3_IMPORTANT_INFO_MISSED": "⚠️",
            "CASE_4_CORRECT_ANSWER_PRODUCED": "🎯",
            "CASE_5_HALLUCINATION_DESPITE_CONTEXT": "🚨",
        }

        for case in trace_data.get("case_studies", []):
            icon = case_icons.get(case["case_id"], "🔍")
            st.markdown(f"#### {icon} {case['title']}")
            st.caption(f"**Diagnostic Focus:** {case['phenomenon']}")

            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("##### 📥 Retrieved Context")
                sim_score = case["highest_similarity"]
                sim_color = "🟢" if sim_score > 0.7 else "🟡" if sim_score > 0.5 else "🔴"
                st.markdown(
                    f"**Max Similarity:** {sim_color} `{sim_score:.4f}` | "
                    f"**Chunks:** `{case['retrieved_count']}`"
                )
                for ex in case.get("retrieved_excerpts", []):
                    with st.expander(f"#{ex['rank']} {ex['document']} — score: `{ex['similarity_score']}`"):
                        st.caption(ex["text_snippet"])

            with c2:
                st.markdown("##### 📤 LLM Response")
                st.markdown(
                    f"**Model:** `{case['model_used']}` | "
                    f"**Gen Time:** `{case['timings']['generation_time_ms']:.0f}` ms | "
                    f"**Embed:** `{case['timings']['embedding_time_ms']:.0f}` ms"
                )
                if case["is_low_confidence"]:
                    st.warning(case["llm_response"])
                else:
                    st.success(case["llm_response"][:500] + ("..." if len(case["llm_response"]) > 500 else ""))

            st.markdown("---")

        # Analysis summary
        st.markdown("#### 📊 Retrieval Quality → Response Quality Analysis")
        case_summaries = []
        for case in trace_data.get("case_studies", []):
            icon = case_icons.get(case["case_id"], "🔍")
            case_summaries.append({
                "Case": f"{icon} {case['case_id']}",
                "Query Type": case.get("title", "")[:50],
                "Max Similarity": case["highest_similarity"],
                "Chunks Retrieved": case["retrieved_count"],
                "Low Confidence": "Yes" if case["is_low_confidence"] else "No",
                "Gen Time (ms)": round(case["timings"]["generation_time_ms"], 0),
            })
        st.dataframe(pd.DataFrame(case_summaries), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Repository Understanding (Exercise 6)
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🏗️ Exercise 6: Repository & Codebase Understanding")
    st.markdown(
        "Investigating whether LLM + RAG can answer questions requiring understanding "
        "of multiple files, module interactions, and architectural ripple effects."
    )

    with st.expander("ℹ️ Exercise 6 — Methodology", expanded=False):
        st.markdown("""
        Investigates 5 repository-level architectural questions across all 3 models:
        1. **Multi-file interaction** — Which files handle query processing, retrieval, generation, telemetry?
        2. **Call-graph dependency** — Which function calls the embedder and vector store during indexing?
        3. **Lifecycle tracing** — What happens after a user uploads a PDF (full 8-step flow)?
        4. **Impact analysis** — If EMBEDDING_DIMENSION changes 384→768, which components break?
        5. **Test traceability** — Which test files validate vector search, chunking, RAG retrieval?

        For each question, measures **Component Recall** = fraction of ground-truth components correctly identified.
        Contrasts chunk-based RAG limitations with Sourcegraph / semantic code-graph paradigms.
        """)

    run_repo_col, _ = st.columns([1, 3])
    with run_repo_col:
        run_repo = st.button(
            "🏗️ Run Repo Understanding",
            help="Runs 5 cross-file questions across all 3 models. Takes ~5-10 min.",
        )

    if run_repo:
        with st.spinner("Running repository understanding investigation (5 questions × 3 models)..."):
            result = subprocess.run(
                [sys.executable, str(_project_root / "evaluation" / "repo_understanding.py")],
                cwd=str(_project_root),
                capture_output=True,
                text=True,
                timeout=1800,
            )
        if result.returncode == 0:
            st.success("✅ Repository understanding investigation complete!")
            st.rerun()
        else:
            st.error(f"❌ Repo understanding failed:\n```\n{result.stderr[-1000:]}\n```")

    if not REPO_FILE.exists():
        st.warning("Repository understanding data not yet generated. Click **🏗️ Run Repo Understanding** above.")
    else:
        with open(REPO_FILE, "r", encoding="utf-8") as f:
            repo_data = json.load(f)

        st.markdown("#### 🧭 Cross-Component Investigation Results")
        for inv in repo_data.get("investigations", []):
            cat_icon = {"multi_file_interaction": "🔗", "call_graph_dependency": "📞",
                       "lifecycle_tracing": "🔄", "impact_analysis": "💥", "test_traceability": "🧪"}.get(inv.get("category", ""), "🔍")
            with st.expander(f"{cat_icon} **[{inv['id']}]** {inv['question'][:70]}..."):
                st.markdown(f"**Category:** `{inv.get('category', '?')}` | **Scope:** `{inv['multi_file_scope']}` files/modules")
                st.markdown(f"**Full Question:** {inv['question']}")

                st.markdown("**Expected Ground-Truth Components:**")
                for comp in inv["ground_truth_components"]:
                    st.markdown(f"- `{comp}`")

                st.markdown("##### Model Recall Performance:")
                model_icons = {"llama3.2:1b": "🦙", "starcoder2:3b": "⭐", "qwen2.5-coder:1.5b": "🧠"}
                m_cols = st.columns(3)
                for idx, (m_name, m_val) in enumerate(inv["model_evaluations"].items()):
                    with m_cols[idx % 3]:
                        icon = model_icons.get(m_name, "🤖")
                        recall_pct = m_val["component_recall"] * 100
                        st.markdown(f"**{icon} {m_name}**")
                        st.metric("Component Recall", f"{recall_pct:.1f}%")
                        st.caption(
                            f"Found {m_val['components_identified']} / "
                            f"{m_val['total_components']} components | "
                            f"Latency: {m_val['latency_ms']:.0f} ms"
                        )
                        with st.expander("View Answer"):
                            st.info(m_val["answer"][:400] + ("..." if len(m_val["answer"]) > 400 else ""))

        # Aggregate recall summary
        st.markdown("---")
        st.markdown("#### 📊 Overall Component Recall by Model")
        model_names_repo = list(repo_data["investigations"][0]["model_evaluations"].keys()) if repo_data.get("investigations") else []
        if model_names_repo:
            recall_rows = []
            for inv in repo_data.get("investigations", []):
                row = {"Investigation": inv["id"], "Category": inv.get("category", "?")}
                for m_name, m_val in inv["model_evaluations"].items():
                    row[m_name] = round(m_val["component_recall"] * 100, 1)
                recall_rows.append(row)
            recall_df = pd.DataFrame(recall_rows).set_index("Investigation")
            st.dataframe(recall_df, use_container_width=True)

            chart_df = pd.DataFrame({
                m: {inv["id"]: inv["model_evaluations"].get(m, {}).get("component_recall", 0) * 100
                    for inv in repo_data.get("investigations", [])}
                for m in model_names_repo
            })
            st.bar_chart(chart_df)

        st.markdown("---")
        st.markdown("#### ⚡ Chunk-Based RAG Limitations vs Code Graph Intelligence")
        arch = repo_data.get("architectural_analysis", {})

        c_lim, c_src = st.columns(2)
        with c_lim:
            st.error("❌ **Limitations of Naive Chunk-Based RAG**")
            for item in arch.get("limitations_of_naive_chunk_rag", []):
                with st.expander(f"**{item['limitation']}**"):
                    st.markdown(item["description"])

        with c_src:
            st.success("✅ **Code Graph Intelligence Advantages (e.g., Sourcegraph)**")
            for item in arch.get("sourcegraph_and_code_graph_advantages", []):
                with st.expander(f"**{item['advantage']}**"):
                    st.markdown(item["description"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: Quick LLM Test
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### ⚡ Quick LLM Response Test")
    st.markdown("Test all 3 models directly with a custom prompt to verify they are responding correctly.")

    ollama_client = get_ollama_client()
    all_models = ollama_client.list_models()

    test_prompt = st.text_area(
        "Test Prompt",
        value="Explain in 2 sentences what a RAG pipeline is and how it improves LLM accuracy.",
        height=80,
    )

    if st.button("🚀 Test All Models", type="primary"):
        if not all_models:
            st.error("No models available. Check Ollama connection.")
        else:
            result_cols = st.columns(len(all_models))
            for i, m in enumerate(all_models):
                m_name = m.get("name", "?")
                with result_cols[i]:
                    st.markdown(f"#### 🤖 `{m_name}`")
                    with st.spinner(f"Generating..."):
                        try:
                            ans, latency = ollama_client.generate(prompt=test_prompt, model=m_name)
                            st.success(f"✅ Responded in `{latency:.0f}` ms")
                            st.info(ans)
                        except Exception as e:
                            st.error(f"❌ Failed: {e}")

    st.markdown("---")
    st.markdown("### 📋 Model Inventory")
    if all_models:
        model_table = []
        for m in all_models:
            model_table.append({
                "Model": m.get("name"),
                "Parameter Size": m.get("details", {}).get("parameter_size", "?"),
                "Quantization": m.get("details", {}).get("quantization_level", "?"),
                "Family": m.get("details", {}).get("family", "?"),
                "Disk Size (MB)": m.get("size", 0) // (1024 * 1024),
            })
        st.dataframe(pd.DataFrame(model_table), use_container_width=True)
    else:
        st.warning("No models detected from Ollama.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6: Guardrails & AI Output Testing
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown("### 🛡️ AI Output & Guardrail Verification Suite")
    st.markdown(
        "Evaluates the multi-layer defensive perimeter protecting the RAG application: "
        "**Input Guardrail**, **Evidence / RAG Guardrail**, and **Untrusted LLM Output Guardrail**."
    )

    with st.expander("ℹ️ Guardrail Architecture & Specification", expanded=False):
        st.markdown(r"""
        | Guardrail Layer | Role & Enforced Rules | Rejection / Action Behavior |
        | :--- | :--- | :--- |
        | **Layer 1: Input Guardrail** | • Rejects empty & whitespace-only queries<br>• Enforces configurable length limit ($\le 500$ chars)<br>• Rejects control characters & null bytes | Blocks Qdrant search & LLM generation entirely.<br>Displays standard error: *"Query too long. Please shorten your question to 500 characters or fewer."* |
        | **Layer 2: Evidence Guardrail** | • Gating: verifies genuine factual evidence beyond raw cosine similarity<br>• Gating on specific entity & topic mismatch (e.g. C vs C++)<br>• Checks target focus terms (e.g. costs/dollars absent from docs) | **Grounded RAG:** Refuses with *"Information not found in the uploaded documents."* without calling LLM.<br>**RAG+LLM:** Falls back to general knowledge with *"General LLM Answer"*. |
        | **Layer 3: Output Guardrail** | • Treats all LLM output as untrusted<br>• Safely strips model tokens (`<end_of_text>`, `<|im_end|>`)<br>• Detects & strips prompt leakage (`System:`, `Context:`)<br>• Enforces grounding consistency | Sanitizes or replaces invalid output with standard notification: *"The generated response could not be validated against the application's requirements."* |
        """)

    # ── Runner trigger ──
    col_gr1, col_gr2 = st.columns([2, 1])
    with col_gr1:
        if st.button("🚀 Run AI Output & Guardrail Benchmark", type="primary"):
            with st.spinner("Executing 22 test cases across all 15 categories..."):
                try:
                    from evaluation.guardrail_runner import GuardrailRunner
                    runner = GuardrailRunner()
                    gr_summary = runner.run()
                    st.success(f"✅ Benchmark executed successfully! Pass Rate: {gr_summary['pass_rate_pct']}%")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error running benchmark: {exc}")

    # ── Load results if available ──
    if GUARDRAILS_RESULTS_FILE.exists():
        with open(GUARDRAILS_RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Tests", data.get("total_tests", 0))
        m2.metric("Passed Tests", data.get("passed", 0))
        m3.metric("Failed Tests", data.get("failed", 0))
        m4.metric("Pass Rate", f"{data.get('pass_rate_pct', 0.0)}%")

        st.markdown("#### 🛡️ Layer Breakdown")
        layers = data.get("layers", {})
        l1, l2, l3 = st.columns(3)
        inp_l = layers.get("input", {})
        ev_l = layers.get("evidence", {})
        out_l = layers.get("output", {})

        l1.metric("Input Guardrail", f"{inp_l.get('passed', 0)} / {inp_l.get('total', 0)}")
        l2.metric("Evidence Guardrail", f"{ev_l.get('passed', 0)} / {ev_l.get('total', 0)}")
        l3.metric("Output Guardrail", f"{out_l.get('passed', 0)} / {out_l.get('total', 0)}")

        st.markdown("---")
        st.markdown("#### 📋 Detailed Test Case Verification (15 Categories)")

        test_cases = data.get("results", [])
        if test_cases:
            df_cases = pd.DataFrame(test_cases)
            cat_filter = st.selectbox(
                "Filter by Category",
                options=["All"] + sorted(list(set(df_cases["category"].unique()))),
            )
            if cat_filter != "All":
                df_cases = df_cases[df_cases["category"] == cat_filter]

            # Style table
            def _style_pass(val):
                color = "#28a745" if val else "#dc3545"
                return f"color: {color}; font-weight: bold"

            st.dataframe(
                df_cases[
                    [
                        "test_id",
                        "category",
                        "query",
                        "target_guardrail",
                        "passed",
                        "actual_triggered",
                        "actual_result_type",
                        "notes",
                    ]
                ],
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("### ⚖️ Before vs After Guardrails Demonstration")
        st.caption("Demonstrating uncontrolled behavior (Without Guardrail) vs controlled behavior (With Guardrail):")

        demo_table = [
            {
                "Test Scenario": "Irrelevant query matching biomedical docs weakly ('which is better c or c++?')",
                "Without Guardrail (Problematic)": "Raw cosine similarity above threshold (0.5656 >= 0.55). Grounded RAG calls LLM with irrelevant chunks and mislabels answer as 'Grounded Document Answer'.",
                "With Guardrail (Controlled)": "Evidence Guardrail evaluates topical relevance, detects mismatch, blocks LLM call (0 tokens), and strictly outputs: 'Information not found in the uploaded documents.'",
            },
            {
                "Test Scenario": "Excessive input query (>500 characters)",
                "Without Guardrail (Problematic)": "Entire 500+ character string is sent to embedding model and Qdrant vector database, wasting computational resources.",
                "With Guardrail (Controlled)": "Input Guardrail rejects query instantly before embedding or retrieval: 'Query too long. Please shorten your question to 500 characters or fewer.'",
            },
            {
                "Test Scenario": "Model generation emitting special tokens ('<end_of_text>', '<|im_end|>')",
                "Without Guardrail (Problematic)": "Raw tokenizer special tokens and instruction artifacts leak into the user's visible answer.",
                "With Guardrail (Controlled)": "Output Guardrail strips all special tokens and prompt header artifacts cleanly before rendering.",
            },
            {
                "Test Scenario": "General knowledge query in RAG+LLM mode ('Who is the Prime Minister of Canada?')",
                "Without Guardrail (Problematic)": "Passes irrelevant document chunks to LLM or fails to distinguish context source.",
                "With Guardrail (Controlled)": "Evidence Guardrail identifies insufficient evidence, falls back cleanly to LLM general knowledge, and explicitly labels it 'General LLM Answer'.",
            },
        ]
        st.table(pd.DataFrame(demo_table))
    else:
        st.info("Guardrail benchmark has not been run yet. Click the button above to execute all 22 test cases.")

