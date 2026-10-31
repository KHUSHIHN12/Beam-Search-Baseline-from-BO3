"""cli.py - Command-line interface for BeamPass compiler optimization."""

from __future__ import annotations

import math
import os
import sys
import tempfile
from typing import Optional

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), ".beampass_mpl")

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from beampass.beam_search import BeamNode, BeamSearch, BeamSearchResult
from beampass.cache import EvaluationCache
from beampass.evaluator import Evaluator, IREEEvaluator, MockEvaluator
from beampass.legality import IRInfo
from beampass.passes import get_default_vocabulary
from beampass.random_search import RandomSearch, RandomSearchResult
from beampass.results import save_run_results
from scripts.export_models import export_and_import
from scripts.verify_passes import verify_all_passes

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure matplotlib has a writable config dir
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(os.path.expanduser("~"), ".beampass_mpl")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

console = Console(force_terminal=True)


def create_evaluator(
    model_name: str,
    mock: bool,
    seed: int = 42,
    cache: Optional[EvaluationCache] = None,
) -> Evaluator:
    """Instantiates either IREEEvaluator or MockEvaluator."""
    if mock:
        return MockEvaluator(model_name=model_name, seed=seed, cache=cache)

    # For Real evaluator, locate or generate MLIR model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlir_path = os.path.join(base_dir, "models", f"{model_name}.mlir")

    if not os.path.exists(mlir_path):
        console.print(f"[yellow][!] Model MLIR not found at {mlir_path}. Attempting export...[/yellow]")
        _, imported_mlir = export_and_import(model_name, output_dir=os.path.join(base_dir, "models"))
        if imported_mlir and os.path.exists(imported_mlir):
            mlir_path = imported_mlir
        else:
            console.print("[red][!] IREE compilation unavailable. Falling back to MockEvaluator (SIMULATED).[/red]")
            return MockEvaluator(model_name=model_name, seed=seed, cache=cache)

    evaluator = IREEEvaluator(
        model_name=model_name,
        model_mlir=mlir_path,
        cache=cache,
    )
    if not evaluator.compiler.is_available() or not evaluator.benchmark.is_available():
        console.print("[yellow][!] IREE compiler or benchmark tool not found. Falling back to MockEvaluator (SIMULATED).[/yellow]")
        return MockEvaluator(model_name=model_name, seed=seed, cache=cache)

    return evaluator


@click.group()
def cli():
    """BeamPass - Beam-Search Compiler Pass-Order Optimizer for IREE."""
    pass


@cli.command()
@click.option("--model", default="small_cnn", help="Target model: small_cnn, mlp, resnet18, mobilenet_v2")
@click.option("--beam-width", "-b", default=2, type=int, help="Beam width B (default: 2)")
@click.option("--depth", "-d", default=5, type=int, help="Maximum search depth (default: 5)")
@click.option("--patience", default=2, type=int, help="Early stopping patience (default: 2)")
@click.option("--max-compiles", default=60, type=int, help="Maximum compilation budget")
@click.option("--mock", is_flag=True, help="Force simulated latency model (SIMULATED)")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--save/--no-save", default=True, help="Save run results to disk")
def run(
    model: str,
    beam_width: int,
    depth: int,
    patience: int,
    max_compiles: int,
    mock: bool,
    seed: int,
    save: bool,
):
    """Run beam search pass-order optimization on a model."""
    console.print(Panel.fit(
        f"[bold cyan]BeamPass Optimizer[/bold cyan]\n"
        f"Model: [green]{model}[/green] | Beam Width: [green]{beam_width}[/green] | Max Depth: [green]{depth}[/green] | Mode: [yellow]{'SIMULATED' if mock else 'AUTO/REAL'}[/yellow]",
        border_style="cyan"
    ))

    cache = EvaluationCache()
    evaluator = create_evaluator(model, mock=mock, seed=seed, cache=cache)
    mode_str = evaluator.mode

    # Determine IR info for structural preconditions
    ir_info = IRInfo(has_conv=(model.lower() != "mlp"))

    # Table for live depth exploration
    current_depth_table: Optional[Table] = None
    depth_nodes: list[BeamNode] = []

    def on_progress(event: dict):
        nonlocal current_depth_table, depth_nodes
        ev = event.get("event")

        if ev == "start":
            console.print(f"[*] Established [bold red]-O3 Baseline[/bold red]: [bold]{event['baseline_latency']:.2f} ms[/bold] [{mode_str}]")
            console.print(f"[*] Beginning Beam Search (Beam Width = {beam_width}, Max Depth = {depth})...\n")

        elif ev == "evaluating":
            seq_str = " -> ".join(event["sequence"])
            console.print(f"  [{event['compile_idx']}/{event['max_compiles']}] Probing Depth {event['depth']}: {seq_str}", end="\r")

        elif ev == "node_evaluated":
            depth_nodes.append(event["node"])

        elif ev == "depth_finished":
            d = event["depth"]
            table = Table(title=f"Depth {d} Candidates (Top {beam_width} kept in beam)", show_header=True, header_style="bold magenta")
            table.add_column("Sequence", style="dim", width=42)
            table.add_column("Latency (ms)", justify="right", width=14)
            table.add_column("Speedup", justify="right", width=12)
            table.add_column("Kept in Beam?", justify="center", width=14)

            for n in depth_nodes:
                kept = "[bold green][KEPT][/bold green]" if n.kept_in_beam else "[dim]pruned[/dim]"
                lat_str = f"{n.latency:.2f}" if not math.isinf(n.latency) else "FAILED (inf)"
                speedup_str = f"{n.speedup_vs_baseline:.2f}x" if n.speedup_vs_baseline > 0 else "-"
                table.add_row(n.sequence_str, lat_str, speedup_str, kept)

            console.print(table)
            depth_nodes.clear()
            console.print(f"[bold cyan]-> Depth {d} Best:[/bold cyan] {event['depth_best_sequence']} ({event['depth_best_latency']:.2f} ms)\n")

    searcher = BeamSearch(
        evaluator=evaluator,
        ir_info=ir_info,
        beam_width=beam_width,
        max_depth=depth,
        patience=patience,
        max_compiles=max_compiles,
        progress_callback=on_progress,
    )

    result = searcher.run()

    # Final summary panel
    seq_display = " -> ".join(result.best_sequence) if result.best_sequence else "(empty: -O3 default)"
    summary = (
        f"[bold]Target Model:[/bold] {result.model_name} [{result.mode}]\n"
        f"[bold]-O3 Baseline Latency:[/bold] [red]{result.baseline_latency:.2f} ms[/red]\n"
        f"[bold]Optimized Latency:[/bold]   [green]{result.final_latency:.2f} ± {result.final_std:.2f} ms[/green]\n"
        f"[bold]Achieved Speedup:[/bold]    [bold yellow]{result.speedup:.2f}x[/bold yellow]\n"
        f"[bold]Total Compilations:[/bold]  {result.total_compiles}\n"
        f"[bold]Best Pass Order:[/bold]     [bold green]{seq_display}[/bold green]"
    )
    console.print(Panel(summary, title="[bold green]BeamPass Optimization Complete[/bold green]", border_style="green"))

    if save:
        saved_paths = save_run_results(result)
        console.print(f"[dim]Results saved: {saved_paths['json']} | Plot: {saved_paths['depth_plot']}[/dim]")


@cli.command()
@click.option("--model", default="small_cnn", help="Target model: small_cnn, mlp, resnet18, mobilenet_v2")
@click.option("--beam-width", "-b", default=2, type=int, help="Beam width B (default: 2)")
@click.option("--depth", "-d", default=5, type=int, help="Maximum search depth (default: 5)")
@click.option("--mock", is_flag=True, help="Force simulated latency model")
@click.option("--seed", default=42, type=int, help="Random seed")
def compare(model: str, beam_width: int, depth: int, mock: bool, seed: int):
    """Compare Beam Search against Random Baseline and -O3 Default (Paper Exp 2a)."""
    console.print(Panel.fit(
        f"[bold cyan]BeamPass Optimizer: Baseline Comparison[/bold cyan]\n"
        f"Model: [green]{model}[/green] | Beam Width: [green]{beam_width}[/green] | Depth: [green]{depth}[/green]",
        border_style="cyan"
    ))

    cache = EvaluationCache()
    evaluator = create_evaluator(model, mock=mock, seed=seed, cache=cache)
    ir_info = IRInfo(has_conv=(model.lower() != "mlp"))

    # 1. Beam Search
    console.print("[*] Running Beam Search...")
    beam_opt = BeamSearch(
        evaluator=evaluator,
        ir_info=ir_info,
        beam_width=beam_width,
        max_depth=depth,
    )
    beam_res = beam_opt.run()

    # 2. Random Search under matched compile budget
    compile_budget = max(beam_res.total_compiles - 1, 10)
    console.print(f"[*] Running Random Rollouts with matched budget ({compile_budget} samples)...")
    rand_opt = RandomSearch(
        evaluator=evaluator,
        ir_info=ir_info,
        max_depth=depth,
        num_samples=compile_budget,
        seed=seed,
    )
    rand_res = rand_opt.run()

    # Comparison Table
    table = Table(title=f"Evaluation Comparison ({model}) [{evaluator.mode}]", show_header=True, header_style="bold cyan")
    table.add_column("Strategy", style="bold", width=18)
    table.add_column("Latency (ms)", justify="right", width=16)
    table.add_column("Speedup vs -O3", justify="right", width=16)
    table.add_column("Compiles", justify="right", width=12)
    table.add_column("Best Pass Sequence", width=36)

    table.add_row(
        "-O3 Baseline",
        f"{beam_res.baseline_latency:.2f}",
        "1.00x",
        "1",
        "(default pipeline)",
    )
    rand_seq = " -> ".join(rand_res.best_sequence) if rand_res.best_sequence else "(none)"
    table.add_row(
        "Random Search",
        f"{rand_res.final_latency:.2f} ± {rand_res.final_std:.2f}",
        f"{rand_res.speedup:.2f}x",
        str(rand_res.total_compiles),
        rand_seq,
    )
    beam_seq = " -> ".join(beam_res.best_sequence) if beam_res.best_sequence else "(none)"
    table.add_row(
        "[bold green]Beam Search[/bold green]",
        f"[bold green]{beam_res.final_latency:.2f} ± {beam_res.final_std:.2f}[/bold green]",
        f"[bold yellow]{beam_res.speedup:.2f}x[/bold yellow]",
        str(beam_res.total_compiles),
        f"[bold green]{beam_seq}[/bold green]",
    )

    console.print(table)
    saved = save_run_results(beam_res, rand_res, tag="compare")
    console.print(f"\n[dim]Comparison artifacts saved: {saved.get('comparison_plot')}[/dim]")


@cli.command("verify-passes")
def cmd_verify_passes():
    """Verify supported IREE preprocessing passes against installed compiler."""
    verify_all_passes(verbose=True)


if __name__ == "__main__":
    cli()
