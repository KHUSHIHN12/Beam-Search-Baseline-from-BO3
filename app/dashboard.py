"""dashboard.py - Interactive Streamlit Demo Dashboard for BeamPass."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add workspace root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from beampass.beam_search import BeamNode, BeamSearch, BeamSearchResult
from beampass.cache import EvaluationCache
from beampass.evaluator import Evaluator, IREEEvaluator, MockEvaluator
from beampass.legality import IRInfo
from beampass.passes import get_default_vocabulary
from beampass.random_search import RandomSearch, RandomSearchResult
from beampass.results import save_run_results


# Page setup
st.set_page_config(
    page_title="BeamPass | IREE Pass-Order Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stMetric {
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #3b82f6;
    }
    .badge-sim {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-real {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_saved_results() -> List[str]:
    """Scans results/ directory for saved json experiments."""
    res_dir = os.path.join(BASE_DIR, "results")
    if not os.path.exists(res_dir):
        return []
    files = [f for f in os.listdir(res_dir) if f.endswith(".json") and f != "cache.json"]
    return sorted(files, reverse=True)


def create_beam_tree_figure(nodes: List[BeamNode], baseline_latency: float) -> go.Figure:
    """Builds an interactive Plotly tree diagram of all explored nodes."""
    fig = go.Figure()

    # Create mapping of sequence tuple to (depth, latency)
    coords: Dict[tuple, tuple] = {}
    for n in nodes:
        seq_key = tuple(n.sequence)
        coords[seq_key] = (n.depth, n.latency)

    # 1. Draw connecting tree edges from parent to child
    edge_x = []
    edge_y = []
    for n in nodes:
        if n.parent_sequence or n.depth > 0:
            parent_key = tuple(n.parent_sequence)
            if parent_key in coords:
                p_depth, p_lat = coords[parent_key]
                edge_x.extend([p_depth, n.depth, None])
                edge_y.extend([p_lat, n.latency, None])

    if edge_x:
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="#cbd5e1", width=1.5),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # 2. Add Pruned nodes
    pruned = [n for n in nodes if not n.kept_in_beam and n.depth > 0 and not math.isinf(n.latency)]
    if pruned:
        fig.add_trace(
            go.Scatter(
                x=[n.depth for n in pruned],
                y=[n.latency for n in pruned],
                mode="markers",
                marker=dict(
                    size=10,
                    color="#94a3b8",
                    symbol="circle",
                    opacity=0.7,
                    line=dict(color="#475569", width=1),
                ),
                text=[f"Seq: {n.sequence_str}<br>Latency: {n.latency:.2f} ms<br>Speedup: {n.speedup_vs_baseline:.2f}x<br>Status: Pruned" for n in pruned],
                hoverinfo="text",
                name="Pruned Candidate",
            )
        )

    # 3. Add Kept in Beam nodes
    kept = [n for n in nodes if n.kept_in_beam and not math.isinf(n.latency)]
    if kept:
        fig.add_trace(
            go.Scatter(
                x=[n.depth for n in kept],
                y=[n.latency for n in kept],
                mode="markers+text",
                marker=dict(
                    size=14,
                    color="#10b981",
                    symbol="diamond",
                    line=dict(color="#064e3b", width=2),
                ),
                text=[n.sequence[-1] if n.sequence else "ROOT" for n in kept],
                textposition="top center",
                textfont=dict(size=10, color="#047857"),
                hovertext=[f"<b>Kept in Beam</b><br>Seq: {n.sequence_str}<br>Latency: {n.latency:.2f} ms<br>Speedup: {n.speedup_vs_baseline:.2f}x" for n in kept],
                hoverinfo="text",
                name="Kept in Beam (Top B)",
            )
        )

    # 4. Add Baseline reference line
    fig.add_hline(
        y=baseline_latency,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"-O3 Baseline ({baseline_latency:.2f} ms)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title="<b>Beam Search Tree Exploration</b> (Green = Kept in Beam, Gray = Pruned)",
        xaxis=dict(title="Search Depth", tickmode="linear", dtick=1),
        yaxis=dict(title="Latency (ms) [Lower is Better]"),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40),
        height=450,
    )
    return fig


def create_comparison_figure(
    baseline_lat: float,
    rand_lat: float,
    beam_lat: float,
    model_name: str,
) -> go.Figure:
    """Creates side-by-side comparison bar chart."""
    categories = ["-O3 Baseline", "Random Search", "Beam Search"]
    latencies = [baseline_lat, rand_lat, beam_lat]
    speedups = [1.0, baseline_lat / rand_lat if rand_lat > 0 else 1.0, baseline_lat / beam_lat if beam_lat > 0 else 1.0]
    colors = ["#94a3b8", "#f59e0b", "#10b981"]

    text_labels = [f"{lat:.2f} ms<br>({sp:.2f}x)" for lat, sp in zip(latencies, speedups)]

    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=latencies,
                text=text_labels,
                textposition="auto",
                marker_color=colors,
                width=0.45,
            )
        ]
    )

    fig.update_layout(
        title=f"<b>Performance Comparison: {model_name}</b>",
        yaxis_title="Latency (ms) [Lower is Better]",
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
    )
    return fig


def create_depth_curve_figure(depth_best: Dict[int, float], baseline_lat: float) -> go.Figure:
    """Creates line chart showing latency improvement with search depth."""
    depths = sorted(depth_best.keys())
    lats = [depth_best[d] for d in depths]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=depths,
            y=lats,
            mode="lines+markers",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=8, color="#1d4ed8"),
            name="Best Latency at Depth",
        )
    )
    fig.add_hline(
        y=baseline_lat,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"-O3 Baseline: {baseline_lat:.2f} ms",
    )
    fig.update_layout(
        title="<b>Best Latency vs Search Depth</b>",
        xaxis=dict(title="Depth", tickmode="linear", dtick=1),
        yaxis=dict(title="Latency (ms)"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
    )
    return fig


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("⚡ BeamPass Controls")
st.sidebar.markdown("Compiler Pass-Order Optimizer for IREE")
st.sidebar.divider()

saved_files = load_saved_results()
selected_saved = None
if saved_files:
    load_choice = st.sidebar.selectbox("📂 Load Pre-recorded Run:", ["(None - Run New Optimization)"] + saved_files)
    if load_choice != "(None - Run New Optimization)":
        selected_saved = os.path.join(BASE_DIR, "results", load_choice)

model_choice = st.sidebar.selectbox(
    "Target Model Architecture:",
    ["small_cnn", "resnet18", "mobilenet_v2", "mlp"],
    index=0,
    help="small_cnn: 3-layer CNN for fast demo (<5s). mlp: Dense net testing legality mask.",
)

beam_width = st.sidebar.slider("Beam Width (B):", min_value=1, max_value=4, value=2, help="Paper default B = 2")
max_depth = st.sidebar.slider("Maximum Search Depth:", min_value=1, max_value=6, value=4)
patience = st.sidebar.slider("Early Stopping Patience:", min_value=1, max_value=3, value=2)

execution_mode = st.sidebar.radio(
    "Execution Mode:",
    ["Mock (Fast Simulated)", "Real IREE (Compile + Benchmark)"],
    index=0,
    help="Mock mode is deterministic and works on any OS. Real mode executes iree-compile and iree-benchmark-module.",
)
is_mock = "Mock" in execution_mode

run_button = st.sidebar.button("Run Beam Optimization", type="primary", use_container_width=True)

# State initialization
if "result_data" not in st.session_state:
    st.session_state["result_data"] = None


# ==========================================
# MAIN PAGE HEADER
# ==========================================
col_hdr, col_badge = st.columns([4, 1])
with col_hdr:
    st.title("BeamPass: Beam-Search Pass-Order Optimizer")
    st.caption("Auto-tuning IREE preprocessing pipelines for neural networks against default `-O3`.")
with col_badge:
    badge_cls = "badge-sim" if is_mock else "badge-real"
    badge_txt = "MODE: SIMULATED" if is_mock else "MODE: REAL IREE"
    st.markdown(f"<div style='text-align: right; margin-top: 15px;'><span class='{badge_cls}'>{badge_txt}</span></div>", unsafe_allow_html=True)

st.divider()


# Handle loading pre-recorded result
if selected_saved and not run_button:
    try:
        with open(selected_saved, "r", encoding="utf-8") as f:
            saved_content = json.load(f)
            st.session_state["result_data"] = saved_content
            st.sidebar.success(f"Loaded: {os.path.basename(selected_saved)}")
    except Exception as e:
        st.sidebar.error(f"Failed to load result: {e}")


# ==========================================
# EXECUTION LOGIC
# ==========================================
if run_button:
    cache = EvaluationCache()
    status_box = st.empty()
    progress_bar = st.progress(0.0)

    # Setup Evaluator
    evaluator: Evaluator
    if is_mock:
        evaluator = MockEvaluator(model_name=model_choice, seed=42, cache=cache)
    else:
        mlir_path = os.path.join(BASE_DIR, "models", f"{model_choice}.mlir")
        evaluator = IREEEvaluator(model_name=model_choice, model_mlir=mlir_path, cache=cache)
        if not evaluator.compiler.is_available() or not evaluator.benchmark.is_available():
            st.warning("⚠️ Real IREE binaries not found. Falling back to MockEvaluator (SIMULATED).")
            evaluator = MockEvaluator(model_name=model_choice, seed=42, cache=cache)

    ir_info = IRInfo(has_conv=(model_choice.lower() != "mlp"))

    # Live UI update structures
    live_rows = []
    live_table_holder = st.empty()

    def dashboard_progress(event: dict):
        ev = event.get("event")
        if ev == "start":
            status_box.info(f"⏳ Established baseline -O3: **{event['baseline_latency']:.2f} ms**. Beginning depth exploration...")
        elif ev == "evaluating":
            pct = min(event["compile_idx"] / event["max_compiles"], 1.0)
            progress_bar.progress(pct)
            status_box.markdown(f"**Exploring Depth {event['depth']}**: Testing `{' -> '.join(event['sequence'])}` (Candidate {event['compile_idx']}/{event['max_compiles']})")
        elif ev == "node_evaluated":
            node: BeamNode = event["node"]
            live_rows.append(
                {
                    "Depth": node.depth,
                    "Pass Sequence": node.sequence_str,
                    "Latency (ms)": f"{node.latency:.2f}" if not math.isinf(node.latency) else "FAILED",
                    "Speedup": f"{node.speedup_vs_baseline:.2f}x" if node.speedup_vs_baseline > 0 else "-",
                }
            )
            # Show the last 6 probed candidates live
            live_table_holder.dataframe(pd.DataFrame(live_rows).tail(6), use_container_width=True)

    optimizer = BeamSearch(
        evaluator=evaluator,
        ir_info=ir_info,
        beam_width=beam_width,
        max_depth=max_depth,
        patience=patience,
        max_compiles=50,
        progress_callback=dashboard_progress,
    )

    with st.spinner("Executing Beam Search Optimization..."):
        beam_res = optimizer.run()

    # Also run matching Random Search baseline for comparison
    with st.spinner("Evaluating Random Search baseline with matched compile budget..."):
        rand_opt = RandomSearch(
            evaluator=evaluator,
            ir_info=ir_info,
            max_depth=max_depth,
            num_samples=min(beam_res.total_compiles - 1, 20),
            seed=42,
        )
        rand_res = rand_opt.run()

    progress_bar.progress(1.0)
    status_box.success("✅ BeamPass Optimization Complete!")
    live_table_holder.empty()

    # Save to disk
    save_run_results(beam_res, rand_res)

    st.session_state["result_data"] = {
        "beam_search": beam_res.to_dict(),
        "random_search": rand_res.to_dict(),
    }


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
    best_seq_str = " ➔ ".join(best_seq) if best_seq else "(empty -O3 default)"
    mode_label = b_data.get("mode", "SIMULATED")

    # Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(label="-O3 Baseline Latency", value=f"{baseline_lat:.2f} ms")
    with c2:
        st.metric(
            label="Optimized Latency",
            value=f"{best_lat:.2f} ms",
            delta=f"-{(baseline_lat - best_lat):.2f} ms",
            delta_color="inverse",
        )
    with c3:
        st.metric(label="Overall Speedup", value=f"{speedup:.2f}x", delta=f"+{((speedup - 1.0) * 100):.1f}%")
    with c4:
        st.metric(label="Total Compilations", value=str(b_data.get("total_compiles", 0)))

    st.info(f"🏆 **Optimal Discovered Pass Sequence:** `{best_seq_str}`")

    # Tabs
    tab_tree, tab_charts, tab_table, tab_theory = st.tabs(
        [
            "🌳 Beam Search Tree",
            "📊 Comparison & Trajectory",
            "📋 Candidate Log",
            "💡 Compiler Theory",
        ]
    )

    # Reconstitute BeamNode objects for tree visualization
    raw_nodes = b_data.get("nodes", [])
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
        for n in raw_nodes
    ]

    with tab_tree:
        fig_tree = create_beam_tree_figure(beam_nodes, baseline_lat)
        st.plotly_chart(fig_tree, use_container_width=True)
        st.caption("Each node represents a pass sequence. Diamond green nodes were kept in the top-B beam for expansion; gray circles were pruned. Hover over nodes to inspect sequence details.")

    with tab_charts:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            rand_lat = r_data.get("final_latency", baseline_lat)
            fig_comp = create_comparison_figure(
                baseline_lat=baseline_lat,
                rand_lat=rand_lat,
                beam_lat=best_lat,
                model_name=b_data.get("model_name", "Model"),
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        with col_c2:
            depth_best = {int(k): v for k, v in b_data.get("depth_best", {}).items()}
            fig_curve = create_depth_curve_figure(depth_best, baseline_lat)
            st.plotly_chart(fig_curve, use_container_width=True)

    with tab_table:
        node_records = []
        for n in beam_nodes:
            node_records.append(
                {
                    "Depth": n.depth,
                    "Sequence": n.sequence_str,
                    "Latency (ms)": round(n.latency, 2) if not math.isinf(n.latency) else "FAILED",
                    "Speedup vs -O3": f"{n.speedup_vs_baseline:.2f}x" if n.speedup_vs_baseline > 0 else "-",
                    "Kept in Beam": "✓ KEPT" if n.kept_in_beam else "pruned",
                    "Compile #": n.compile_idx,
                }
            )
        df_nodes = pd.DataFrame(node_records)
        st.dataframe(df_nodes, use_container_width=True, height=350)

    with tab_theory:
        st.markdown(
            """
            ### Why Compiler Pass Ordering Matters
            Modern ML compilers like **IREE** (Intermediate Representation Execution Environment) rely on a sequence of progressive lowering transformations (from high-level tensor operations down to target vector and machine instructions).

            > **Progressive Lowering & Regime Shifts:**
            > Applying a transformation changes the IR structure, opening opportunities for downstream passes while closing others. For instance:
            > - **Channels-Last Conversion (`CC`) ➔ Img2Col (`I`)**: Converting image data layouts from NCHW to NHWC channels-last exposes contiguous memory blocks that enable Img2Col matrix-multiplication lowering to fully exploit hardware SIMD vector units.
            > - In contrast, applying Img2Col without channels-last conversion or applying redundant generalizes can trigger memory fragmentation and unvectorized scalar fallbacks.

            #### Different Models Need Different Pipelines (Paper Fig. 2)
            A static `-O3` pipeline uses a fixed heuristic order for all networks. However:
            1. **Vision Models (ResNet, MobileNet):** Heavily bounded by convolution memory access layouts; benefit most from `CC -> I -> pad` sequences.
            2. **Dense Networks & Transformers (MLP, BERT):** Lack spatial convolutions; benefit from matrix-multiplication operand transposition and intrinsics padding.
            3. **Legality Masking:** Ensures only valid transformations are evaluated, preventing compilation crashes and avoiding wasted search budgets.
            """
        )

else:
    st.info("👈 Configure optimization parameters in the sidebar and click **'Run Beam Optimization'**, or select a pre-recorded run from the dropdown to see instant results.")

