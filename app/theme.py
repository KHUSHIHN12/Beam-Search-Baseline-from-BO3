"""theme.py - Consistent design system and Plotly chart builders for BeamPass."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
from beampass.beam_search import BeamNode


# Color Tokens matching the design system
COLOR_BG = "#FFFFFF"
COLOR_GRID = "#EEECE7"
COLOR_AXIS = "#C9C6BE"
COLOR_TEXT_PRIMARY = "#1C1B19"
COLOR_TEXT_SECONDARY = "#6B6963"
COLOR_TEXT_MUTED = "#9A978F"
COLOR_ACCENT = "#0F766E"         # Deep teal (single accent)
COLOR_ACCENT_TINT = "#E6F2F0"    # Subtle teal background
COLOR_NEUTRAL_LIGHT = "#E4E2DC"
COLOR_NEUTRAL_MID = "#C9C6BE"
COLOR_NEUTRAL_DARK = "#9A978F"
COLOR_ERROR = "#B4412F"          # Muted brick red


def get_base_layout(**kwargs) -> dict:
    """Returns baseline Plotly layout with typography and grid styles."""
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Inter, -apple-system, sans-serif",
            size=12,
            color=COLOR_TEXT_SECONDARY,
        ),
        xaxis=dict(
            gridcolor=COLOR_GRID,
            linecolor=COLOR_AXIS,
            zerolinecolor=COLOR_GRID,
            tickfont=dict(size=11, color=COLOR_TEXT_SECONDARY),
            titlefont=dict(size=12, color=COLOR_TEXT_PRIMARY),
        ),
        yaxis=dict(
            gridcolor=COLOR_GRID,
            linecolor=COLOR_AXIS,
            zerolinecolor=COLOR_GRID,
            tickfont=dict(size=11, color=COLOR_TEXT_SECONDARY),
            titlefont=dict(size=12, color=COLOR_TEXT_PRIMARY),
        ),
        margin=dict(l=48, r=24, t=16, b=40),
        showlegend=False,
    )
    layout.update(kwargs)
    return layout


def build_beam_tree_chart(
    nodes: List[BeamNode],
    baseline_latency: float,
    best_sequence: Optional[List[str]] = None,
) -> go.Figure:
    """Builds the tree chart with monochrome pruned nodes and teal kept nodes."""
    fig = go.Figure()
    best_seq = best_sequence or []

    # Map sequence tuple to (depth, latency)
    coords: Dict[Tuple[str, ...], Tuple[int, float]] = {}
    for n in nodes:
        coords[tuple(n.sequence)] = (n.depth, n.latency)

    # 1. Normal tree edges (light grey)
    edge_x = []
    edge_y = []
    # Best path edges (thick teal)
    best_edge_x = []
    best_edge_y = []

    for n in nodes:
        if n.parent_sequence or n.depth > 0:
            parent_key = tuple(n.parent_sequence)
            if parent_key in coords:
                p_depth, p_lat = coords[parent_key]

                # Check if this edge is along the best sequence path
                is_best_edge = False
                n_len = len(n.sequence)
                if n_len <= len(best_seq) and n.sequence == best_seq[:n_len]:
                    is_best_edge = True

                if is_best_edge:
                    best_edge_x.extend([p_depth, n.depth, None])
                    best_edge_y.extend([p_lat, n.latency, None])
                else:
                    edge_x.extend([p_depth, n.depth, None])
                    edge_y.extend([p_lat, n.latency, None])

    # Draw regular connecting lines
    if edge_x:
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color=COLOR_GRID, width=1.2),
                hoverinfo="none",
            )
        )

    # Draw emphasized best path line
    if best_edge_x:
        fig.add_trace(
            go.Scatter(
                x=best_edge_x,
                y=best_edge_y,
                mode="lines",
                line=dict(color=COLOR_ACCENT, width=2.4),
                hoverinfo="none",
            )
        )

    # 2. Pruned nodes (neutral light grey with thin border)
    pruned = [n for n in nodes if not n.kept_in_beam and n.depth > 0 and not math.isinf(n.latency)]
    if pruned:
        fig.add_trace(
            go.Scatter(
                x=[n.depth for n in pruned],
                y=[n.latency for n in pruned],
                mode="markers",
                marker=dict(
                    size=8,
                    color=COLOR_NEUTRAL_LIGHT,
                    line=dict(color=COLOR_AXIS, width=1),
                ),
                text=[
                    f"Sequence: {n.sequence_str}<br>Latency: {n.latency:.2f} ms<br>Status: Pruned"
                    for n in pruned
                ],
                hoverinfo="text",
            )
        )

    # 3. Kept in Beam nodes (accent teal, small monospace label)
    kept = [n for n in nodes if n.kept_in_beam and not math.isinf(n.latency)]
    if kept:
        fig.add_trace(
            go.Scatter(
                x=[n.depth for n in kept],
                y=[n.latency for n in kept],
                mode="markers+text",
                marker=dict(
                    size=12,
                    color=COLOR_ACCENT,
                    symbol="circle",
                    line=dict(color="#0D645D", width=1.5),
                ),
                text=[n.sequence[-1] if n.sequence else "ROOT" for n in kept],
                textposition="top center",
                textfont=dict(family="JetBrains Mono, monospace", size=10, color=COLOR_ACCENT),
                hovertext=[
                    f"<b>Kept in Beam</b><br>Sequence: {n.sequence_str}<br>Latency: {n.latency:.2f} ms<br>Speedup: {n.speedup_vs_baseline:.2f}x"
                    for n in kept
                ],
                hoverinfo="text",
            )
        )

    # 4. -O3 Baseline horizontal reference line
    fig.add_hline(
        y=baseline_latency,
        line_dash="dash",
        line_color=COLOR_NEUTRAL_DARK,
        line_width=1.2,
        annotation_text=f"-O3 baseline ({baseline_latency:.2f} ms)",
        annotation_position="bottom right",
        annotation_font=dict(family="Inter, sans-serif", size=11, color=COLOR_TEXT_SECONDARY),
    )

    # Dynamic depth column labels
    max_d = max([n.depth for n in nodes], default=1)
    tickvals = list(range(max_d + 1))
    ticktext = ["Baseline" if d == 0 else f"Depth {d}" for d in tickvals]

    layout = get_base_layout(
        xaxis=dict(
            tickvals=tickvals,
            ticktext=ticktext,
            gridcolor=COLOR_GRID,
            linecolor=COLOR_AXIS,
        ),
        yaxis=dict(
            title="Latency (ms)",
            gridcolor=COLOR_GRID,
            linecolor=COLOR_AXIS,
        ),
        height=420,
    )
    fig.update_layout(layout)
    return fig


def build_comparison_bar_chart(
    baseline_lat: float,
    rand_lat: float,
    beam_lat: float,
) -> go.Figure:
    """Builds clean horizontal bar chart with directly labeled ends and no legend."""
    categories = ["-O3 baseline", "Random search", "Beam search"]
    latencies = [baseline_lat, rand_lat, beam_lat]
    colors = [COLOR_NEUTRAL_MID, COLOR_NEUTRAL_DARK, COLOR_ACCENT]

    speedup_rand = baseline_lat / rand_lat if rand_lat > 0 else 1.0
    speedup_beam = baseline_lat / beam_lat if beam_lat > 0 else 1.0

    text_labels = [
        f"{baseline_lat:.2f} ms  (1.00x)",
        f"{rand_lat:.2f} ms  ({speedup_rand:.2f}x)",
        f"{beam_lat:.2f} ms  ({speedup_beam:.2f}x)",
    ]

    fig = go.Figure(
        go.Bar(
            y=categories,
            x=latencies,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=text_labels,
            textposition="outside",
            textfont=dict(family="JetBrains Mono, monospace", size=11, color=COLOR_TEXT_PRIMARY),
            cliponaxis=False,
            width=0.45,
        )
    )

    max_x = max(latencies) * 1.3
    layout = get_base_layout(
        xaxis=dict(
            title="Latency (ms)",
            range=[0, max_x],
            gridcolor=COLOR_GRID,
            linecolor=COLOR_AXIS,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(family="Inter, sans-serif", size=12, color=COLOR_TEXT_PRIMARY),
            linecolor=COLOR_AXIS,
        ),
        margin=dict(l=110, r=70, t=12, b=36),
        height=220,
    )
    fig.update_layout(layout)
    return fig


def build_latency_depth_chart(
    depth_best: Dict[int, float],
    baseline_lat: float,
) -> go.Figure:
    """Builds latency vs depth line chart with direct baseline annotation."""
    depths = sorted(depth_best.keys())
    lats = [depth_best[d] for d in depths]

    fig = go.Figure()

    # Best latency line
    fig.add_trace(
        go.Scatter(
            x=depths,
            y=lats,
            mode="lines+markers",
            line=dict(color=COLOR_ACCENT, width=2),
            marker=dict(size=6, color=COLOR_ACCENT),
            hoverinfo="x+y",
            hovertemplate="Depth %{x}: %{y:.2f} ms<extra></extra>",
        )
    )

    # -O3 Baseline horizontal reference line
    fig.add_hline(
        y=baseline_lat,
        line_dash="dash",
        line_color=COLOR_NEUTRAL_DARK,
        line_width=1.2,
        annotation_text=f"-O3 ({baseline_lat:.2f} ms)",
        annotation_position="bottom right",
        annotation_font=dict(family="Inter, sans-serif", size=11, color=COLOR_TEXT_SECONDARY),
    )

    layout = get_base_layout(
        xaxis=dict(
            title="Search depth",
            tickmode="linear",
            dtick=1,
            gridcolor=COLOR_GRID,
            linecolor=COLOR_AXIS,
        ),
        yaxis=dict(
            title="Latency (ms)",
            gridcolor=COLOR_GRID,
            linecolor=COLOR_AXIS,
        ),
        margin=dict(l=48, r=24, t=12, b=36),
        height=260,
    )
    fig.update_layout(layout)
    return fig
