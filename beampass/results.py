"""results.py - Persistence, export, and plotting for BeamPass search results."""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import time
from typing import Dict, List, Optional, Union

# Set temporary MPLCONFIGDIR before importing matplotlib to prevent warnings
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), ".beampass_mpl")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import pandas as pd

from beampass.beam_search import BeamSearchResult
from beampass.random_search import RandomSearchResult


def save_run_results(
    beam_result: BeamSearchResult,
    random_result: Optional[RandomSearchResult] = None,
    output_dir: str = "results",
    tag: Optional[str] = None,
) -> Dict[str, str]:
    """Saves search results to JSON, CSV, and generates summary plots."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model = beam_result.model_name
    tag_str = f"_{tag}" if tag else ""
    base_name = f"{model}_{timestamp}{tag_str}"

    # 1. Save JSON
    json_path = os.path.join(output_dir, f"{base_name}.json")
    data = {
        "timestamp": timestamp,
        "beam_search": beam_result.to_dict(),
        "random_search": random_result.to_dict() if random_result else None,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # 2. Save CSV of beam nodes
    csv_path = os.path.join(output_dir, f"{base_name}_nodes.csv")
    node_records = []
    for node in beam_result.nodes:
        node_records.append(
            {
                "depth": node.depth,
                "sequence": " -> ".join(node.sequence) if node.sequence else "EMPTY",
                "latency_ms": node.latency,
                "kept_in_beam": node.kept_in_beam,
                "speedup_vs_baseline": node.speedup_vs_baseline,
                "parent_sequence": " -> ".join(node.parent_sequence) if node.parent_sequence else "ROOT",
                "compile_idx": node.compile_idx,
            }
        )
    df = pd.DataFrame(node_records)
    df.to_csv(csv_path, index=False)

    # 3. Generate Latency vs Depth plot
    depth_plot_path = os.path.join(output_dir, f"{base_name}_latency_depth.png")
    plot_latency_vs_depth(beam_result, depth_plot_path)

    # 4. Generate comparison plot if random baseline exists
    comp_plot_path = None
    if random_result:
        comp_plot_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        plot_comparison_bar(beam_result.baseline_latency, random_result, beam_result, comp_plot_path)

    paths = {
        "json": json_path,
        "csv": csv_path,
        "depth_plot": depth_plot_path,
    }
    if comp_plot_path:
        paths["comparison_plot"] = comp_plot_path

    return paths


def load_run_results(json_path: str) -> Dict:
    """Loads a previously saved run result from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_latency_vs_depth(
    beam_result: BeamSearchResult,
    out_path: Optional[str] = None,
) -> None:
    """Generates a line plot of best latency versus search depth."""
    plt.figure(figsize=(7, 4.5), dpi=150)

    depths = sorted(beam_result.depth_best.keys())
    latencies = [beam_result.depth_best[d] for d in depths]

    plt.plot(
        depths,
        latencies,
        marker="o",
        color="#2563EB",
        linewidth=2.5,
        markersize=6,
        label="Beam Search Best",
    )
    plt.axhline(
        y=beam_result.baseline_latency,
        color="#DC2626",
        linestyle="--",
        label=f"-O3 Baseline ({beam_result.baseline_latency:.2f} ms)",
    )

    plt.title(f"BeamPass Latency vs Depth ({beam_result.model_name}) [{beam_result.mode}]", fontsize=12, fontweight="bold")
    plt.xlabel("Search Depth", fontsize=10)
    plt.ylabel("Latency (ms)", fontsize=10)
    plt.xticks(depths)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path)
    plt.close()


def plot_comparison_bar(
    baseline_lat: float,
    random_result: RandomSearchResult,
    beam_result: BeamSearchResult,
    out_path: Optional[str] = None,
) -> None:
    """Generates a comparison bar chart: -O3 vs Random vs Beam Search."""
    plt.figure(figsize=(7, 4.5), dpi=150)

    labels = ["-O3 Baseline", "Random Search", "Beam Search"]
    lats = [baseline_lat, random_result.final_latency, beam_result.final_latency]
    colors = ["#94A3B8", "#F59E0B", "#10B981"]

    bars = plt.bar(labels, lats, color=colors, width=0.55, edgecolor="#1E293B", linewidth=1.2)

    # Add data labels on top of bars
    for bar, lat in zip(bars, lats):
        height = bar.get_height()
        speedup = baseline_lat / lat if lat > 0 else 1.0
        label_text = f"{lat:.2f} ms\n({speedup:.2f}x)"
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(lats) * 0.02,
            label_text,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.ylim(0, max(lats) * 1.25)
    plt.title(f"Compiler Pass Optimizer Comparison ({beam_result.model_name}) [{beam_result.mode}]", fontsize=12, fontweight="bold")
    plt.ylabel("Latency (ms) [Lower is Better]", fontsize=10)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path)
    plt.close()
