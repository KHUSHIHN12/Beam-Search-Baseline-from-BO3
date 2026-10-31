"""dashboard.py - Clean, data-first Streamlit engineering dashboard for BeamPass."""

from __future__ import annotations

import json
import math
import os
import sys
import textwrap
import time
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# Add workspace root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.theme import (
    build_beam_tree_chart,
    build_comparison_bar_chart,
    build_latency_depth_chart,
)
from beampass.beam_search import BeamNode, BeamSearch, BeamSearchResult
from beampass.cache import EvaluationCache
from beampass.evaluator import Evaluator, IREEEvaluator, MockEvaluator
from beampass.legality import IRInfo
from beampass.passes import get_default_vocabulary
from beampass.random_search import RandomSearch, RandomSearchResult
from beampass.results import save_run_results


# Page configuration
st.set_page_config(
    page_title="BeamPass | Compiler Pass Optimizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom design system CSS
styles_path = os.path.join(os.path.dirname(__file__), "styles.css")
if os.path.exists(styles_path):
    with open(styles_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


PASS_FULL_NAMES = {
    "canon": "canonicalize",
    "CC": "iree-preprocessing-convert-conv-to-channels-last",
    "I": "iree-global-opt-convert-conv2d-to-img2col",
    "T": "iree-preprocessing-transpose-convolution-pipeline",
    "gen": "linalg-generalize-named-ops",
    "1x1": "iree-preprocessing-convert-1x1-filter-conv2d-to-matmul",
    "pad": "iree-preprocessing-pad-linalg-ops",
    "intrinsics": "iree-preprocessing-pad-to-intrinsics",
    "trans-mm": "iree-preprocessing-transpose-matmul-pass",
}


def load_saved_runs() -> List[str]:
    """Scans results/ directory for saved run JSON files."""
    res_dir = os.path.join(BASE_DIR, "results")
    if not os.path.exists(res_dir):
        return []
    files = [f for f in os.listdir(res_dir) if f.endswith(".json") and f != "cache.json"]
    return sorted(files, reverse=True)


# Initialize session state
if "result_data" not in st.session_state:
    st.session_state["result_data"] = None

# Default to first pre-recorded run if available and not yet loaded
saved_runs = load_saved_runs()
if st.session_state["result_data"] is None and saved_runs:
    try:
        with open(os.path.join(BASE_DIR, "results", saved_runs[0]), "r", encoding="utf-8") as f:
            st.session_state["result_data"] = json.load(f)
    except Exception:
        pass


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown('<div class="sidebar-section-label">Workload</div>', unsafe_allow_html=True)
model_choice = st.sidebar.selectbox(
    "Target Model Architecture",
    ["small_cnn", "resnet18", "mobilenet_v2", "mlp"],
    index=0,
    format_func=lambda x: {
        "small_cnn": "small_cnn (3-layer CNN)",
        "resnet18": "resnet18 (ResNet-18)",
        "mobilenet_v2": "mobilenet_v2 (MobileNetV2)",
        "mlp": "mlp (Dense MLP, no conv)",
    }.get(x, x),
    label_visibility="collapsed",
)

st.sidebar.markdown('<div class="sidebar-section-label">Search</div>', unsafe_allow_html=True)
beam_width = st.sidebar.slider("Beam width (B)", min_value=1, max_value=4, value=2)
max_depth = st.sidebar.slider("Maximum search depth", min_value=1, max_value=6, value=4)
patience = st.sidebar.slider("Early stopping patience", min_value=1, max_value=3, value=2)

st.sidebar.markdown('<div class="sidebar-section-label">Execution</div>', unsafe_allow_html=True)
execution_mode = st.sidebar.radio(
    "Execution mode",
    ["Simulated (deterministic mock)", "Real IREE (hardware benchmark)"],
    index=0,
    label_visibility="collapsed",
)
is_mock = "Simulated" in execution_mode

run_clicked = st.sidebar.button("Run search", type="primary", use_container_width=True)

st.sidebar.markdown('<div class="sidebar-section-label">Saved Run</div>', unsafe_allow_html=True)
if saved_runs:
    load_choice = st.sidebar.selectbox(
        "Load saved run",
        ["(Current / New run)"] + saved_runs,
        label_visibility="collapsed",
    )
    if load_choice != "(Current / New run)":
        saved_path = os.path.join(BASE_DIR, "results", load_choice)
        try:
            with open(saved_path, "r", encoding="utf-8") as f:
                st.session_state["result_data"] = json.load(f)
        except Exception:
            pass


# ==========================================
# EXECUTION ACTION
# ==========================================
run_duration_str = None
if run_clicked:
    start_time = time.time()
    cache = EvaluationCache()

    evaluator: Evaluator
    if is_mock:
        evaluator = MockEvaluator(model_name=model_choice, seed=42, cache=cache)
    else:
        mlir_path = os.path.join(BASE_DIR, "models", f"{model_choice}.mlir")
        evaluator = IREEEvaluator(model_name=model_choice, model_mlir=mlir_path, cache=cache)
        if not evaluator.compiler.is_available() or not evaluator.benchmark.is_available():
            evaluator = MockEvaluator(model_name=model_choice, seed=42, cache=cache)

    ir_info = IRInfo(has_conv=(model_choice.lower() != "mlp"))

    progress_bar = st.progress(0.0)
    status_line = st.empty()
    live_table_holder = st.empty()
    live_rows = []

    def handle_progress(event: dict):
        ev = event.get("event")
        if ev == "start":
            status_line.markdown(
                f'<div class="quiet-status">Baseline -O3: {event["baseline_latency"]:.2f} ms. Beginning search...</div>',
                unsafe_allow_html=True,
            )
        elif ev == "evaluating":
            pct = min(event["compile_idx"] / event["max_compiles"], 1.0)
            progress_bar.progress(pct)
            seq_display = " > ".join(event["sequence"])
            status_line.markdown(
                f'<div class="quiet-status">Depth {event["depth"]} — candidate {event["compile_idx"]} of {event["max_compiles"]}: {seq_display}</div>',
                unsafe_allow_html=True,
            )
        elif ev == "node_evaluated":
            node: BeamNode = event["node"]
            live_rows.append(
                {
                    "Depth": node.depth,
                    "Sequence": node.sequence_str,
                    "Latency (ms)": f"{node.latency:.2f}" if not math.isinf(node.latency) else "failed",
                    "Speedup": f"{node.speedup_vs_baseline:.2f}x" if node.speedup_vs_baseline > 0 else "-",
                }
            )
            live_table_holder.dataframe(pd.DataFrame(live_rows).tail(5), use_container_width=True)

    searcher = BeamSearch(
        evaluator=evaluator,
        ir_info=ir_info,
        beam_width=beam_width,
        max_depth=max_depth,
        patience=patience,
        max_compiles=50,
        progress_callback=handle_progress,
    )

    beam_res = searcher.run()

    # Matched random baseline
    matched_budget = min(beam_res.total_compiles - 1, 20)
    rand_searcher = RandomSearch(
        evaluator=evaluator,
        ir_info=ir_info,
        max_depth=max_depth,
        num_samples=matched_budget,
        seed=42,
    )
    rand_res = rand_searcher.run()

    elapsed = time.time() - start_time
    mins, secs = int(elapsed // 60), int(elapsed % 60)
    run_duration_str = f"Completed in {mins}m {secs:02d}s" if mins > 0 else f"Completed in {secs}s"

    progress_bar.empty()
    status_line.markdown(
        f'<div class="quiet-success">{run_duration_str} ({beam_res.total_compiles} compilations evaluated)</div>',
        unsafe_allow_html=True,
    )
    live_table_holder.empty()

    save_run_results(beam_res, rand_res)
    st.session_state["result_data"] = {
        "beam_search": beam_res.to_dict(),
        "random_search": rand_res.to_dict(),
        "duration_str": run_duration_str,
    }


# ==========================================
# HEADER BAR
# ==========================================
if is_mock:
    current_mode_is_sim = True
elif st.session_state["result_data"]:
    b_data = st.session_state["result_data"].get("beam_search", {})
    current_mode_is_sim = (b_data.get("mode", "REAL") == "SIMULATED")
else:
    current_mode_is_sim = False

pill_class = "mode-pill-simulated" if current_mode_is_sim else "mode-pill-real"
pill_text = "Simulated" if current_mode_is_sim else "Real IREE"

st.markdown(
    textwrap.dedent(
        f"""
        <div class="top-header">
            <div class="top-title-group">
                <h1 class="top-title">BeamPass</h1>
                <span class="top-subtitle">IREE preprocessing pass-order search</span>
            </div>
            <div>
                <span class="mode-pill {pill_class}">{pill_text}</span>
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)


# ==========================================
# RESULTS DISPLAY
# ==========================================
data = st.session_state.get("result_data")

if data and "beam_search" in data:
    b_data = data["beam_search"]
    r_data = data.get("random_search", {})

    baseline_lat = b_data.get("baseline_latency", 100.0)
    best_lat = b_data.get("final_latency", b_data.get("best_latency", 100.0))
    speedup = b_data.get("speedup", baseline_lat / best_lat if best_lat > 0 else 1.0)
    best_seq = b_data.get("best_sequence", [])
    total_compiles = b_data.get("total_compiles", 0)

    delta_lat = baseline_lat - best_lat
    pct_speedup = (speedup - 1.0) * 100

    # 1. Unified 4-column Results Container
    st.markdown(
        textwrap.dedent(
            f"""
            <div class="metrics-container">
                <div class="metric-cell">
                    <div class="metric-label">-O3 Baseline</div>
                    <div class="metric-value">{baseline_lat:.2f} ms</div>
                    <div class="metric-delta">Default IREE pipeline</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-label">Best Found</div>
                    <div class="metric-value">{best_lat:.2f} ms</div>
                    <div class="metric-delta metric-delta-positive">-{delta_lat:.2f} ms ({pct_speedup:.1f}% reduction)</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-label">Speedup</div>
                    <div class="metric-value metric-value-accent">{speedup:.2f}x</div>
                    <div class="metric-delta metric-delta-positive">Relative to -O3</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-label">Compilations</div>
                    <div class="metric-value">{total_compiles}</div>
                    <div class="metric-delta">Search budget used</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    # 2. Best Sequence Display with Monospace Chips
    chips_html = ""
    if best_seq:
        chip_items = []
        for i, s in enumerate(best_seq):
            chip_items.append(f'<span class="pass-chip">{s}</span>')
            if i < len(best_seq) - 1:
                chip_items.append('<span class="chip-arrow">&rarr;</span>')
        chips_html = "".join(chip_items)
        full_names_str = " &rarr; ".join([PASS_FULL_NAMES.get(s, s) for s in best_seq])
    else:
        chips_html = '<span class="pass-chip">(empty: default -O3)</span>'
        full_names_str = "None (empty sequence fallback)"

    st.markdown(
        textwrap.dedent(
            f"""
            <div class="sequence-container">
                <div class="sequence-label">Optimal Preprocessing Pipeline</div>
                <div class="chip-chain">{chips_html}</div>
                <div class="sequence-caption">{full_names_str}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    # 3. Clean Tabs without emojis
    tab_tree, tab_comparison, tab_candidates, tab_background = st.tabs(
        [
            "Search tree",
            "Comparison",
            "Candidates",
            "Background",
        ]
    )

    # Reconstruct BeamNode objects for plotting
    beam_nodes = [
        BeamNode(
            depth=n["depth"],
            sequence=n["sequence"],
            latency=n["latency"],
            kept_in_beam=n["kept_in_beam"],
            parent_sequence=n.get("parent_sequence", []),
            compile_idx=n.get("compile_idx", 0),
            speedup_vs_baseline=n.get("speedup_vs_baseline", 1.0),
        )
        for n in b_data.get("nodes", [])
    ]

    with tab_tree:
        st.markdown('<div class="chart-heading">Beam search exploration tree</div>', unsafe_allow_html=True)
        fig_tree = build_beam_tree_chart(
            nodes=beam_nodes,
            baseline_latency=baseline_lat,
            best_sequence=best_seq,
        )
        st.plotly_chart(fig_tree, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            '<div class="chart-caption">Evaluated pass sequences by search depth. Teal nodes were retained in the beam (B=2); grey nodes were pruned. The thicker teal line highlights the optimal pass sequence path.</div>',
            unsafe_allow_html=True,
        )

    with tab_comparison:
        st.markdown('<div class="chart-heading">Compiler pass pipeline latency</div>', unsafe_allow_html=True)
        rand_lat = r_data.get("final_latency", baseline_lat)
        fig_bar = build_comparison_bar_chart(
            baseline_lat=baseline_lat,
            rand_lat=rand_lat,
            beam_lat=best_lat,
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            '<div class="chart-caption">End-to-end execution latency on single-core CPU (local-sync device). Shorter bars indicate faster execution.</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="chart-heading">Latency vs search depth</div>', unsafe_allow_html=True)
        depth_best = {int(k): v for k, v in b_data.get("depth_best", {}).items()}
        fig_depth = build_latency_depth_chart(depth_best, baseline_lat)
        st.plotly_chart(fig_depth, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            '<div class="chart-caption">Progressive latency reduction as the beam explores deeper pass sequences against the static -O3 baseline.</div>',
            unsafe_allow_html=True,
        )

    with tab_candidates:
        st.markdown('<div class="chart-heading">Evaluated candidates</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="chart-caption">All explored pass sequences, sorted by search depth and execution latency.</div>',
            unsafe_allow_html=True,
        )

        records = []
        for n in beam_nodes:
            is_failed = math.isinf(n.latency)
            records.append(
                {
                    "Depth": n.depth,
                    "Sequence": " > ".join(n.sequence) if n.sequence else "(baseline)",
                    "Latency": None if is_failed else round(n.latency, 2),
                    "Status": "Failed" if is_failed else ("Kept in beam" if n.kept_in_beam else "Pruned"),
                    "Speedup": None if is_failed else f"{n.speedup_vs_baseline:.2f}x",
                    "Compile #": n.compile_idx,
                }
            )

        df = pd.DataFrame(records)
        df = df.sort_values(by=["Depth", "Latency"], ascending=[True, True])

        st.dataframe(
            df,
            column_config={
                "Latency": st.column_config.NumberColumn("Latency", format="%.2f ms"),
                "Speedup": st.column_config.TextColumn("Speedup vs -O3"),
                "Status": st.column_config.TextColumn("Status"),
                "Depth": st.column_config.NumberColumn("Depth", format="%d"),
            },
            hide_index=True,
            use_container_width=True,
            height=380,
        )

    with tab_background:
        st.markdown(
            """<div class="background-column">
<div class="background-section">
<h3>Why pass order matters</h3>
<p>
Modern ML compilers such as IREE lower high-level tensor operations into executable machine
code through successive Intermediate Representation (IR) transformations. Because each compiler
pass alters the IR structure, applying a pass can expose optimization opportunities or permanently
close downstream lowering paths.
</p>
<p>
For instance, converting convolution layouts from NCHW to NHWC channels-last exposes contiguous
memory dimensions that allow Img2Col lowering to fully utilize CPU SIMD vector execution. Applying
Img2Col without prior layout conversion yields scalar fallbacks and memory fragmentation.
</p>
</div>

<div class="pipeline-diagram">
<div class="pipeline-box pipeline-box-highlight">
<div class="pipeline-box-title">Preprocessing</div>
<div class="pipeline-box-sub">Tuned by BeamPass</div>
</div>
<span class="pipeline-arrow">&rarr;</span>
<div class="pipeline-box">
<div class="pipeline-box-title">Dispatch Creation</div>
<div class="pipeline-box-sub">Kernel partition & fusion</div>
</div>
<span class="pipeline-arrow">&rarr;</span>
<div class="pipeline-box">
<div class="pipeline-box-title">Code Generation</div>
<div class="pipeline-box-sub">LLVM CPU vectorization</div>
</div>
</div>

<div class="background-section">
<h3>What beam search does here</h3>
<p>
The search space of compiler pass sequences grows exponentially with pipeline length. Exhaustive
search is intractable, while greedy search frequently gets trapped in sub-optimal local minima.
</p>
<p>
BeamPass uses beam search with a width of <i>B = 2</i>. At each depth, it evaluates all legal
successors, filters invalid transformations through legality masks (such as masking convolution
passes on dense models), retains only the top <i>B</i> lowest-latency sequences, and prunes the remainder.
</p>
</div>

<div class="background-section">
<h3>How latency is measured</h3>
<p>
Latency measurements follow the paper's multi-tier protocol using <code>iree-benchmark-module</code>:
</p>
<p>
<b>Selection phase:</b> 10 repetitions per candidate; the median is used to rank beam candidates.<br>
<b>Final evaluation:</b> 10 independent invocations with 5 repetitions each; reports the mean &plusmn; standard deviation of the medians.
</p>
</div>
</div>""",
            unsafe_allow_html=True,
        )

else:
    st.info("Configure workload parameters in the sidebar and select 'Run search' to begin optimization.")
