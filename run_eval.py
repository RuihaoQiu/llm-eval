import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import logfire
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text

from llm_eval.eval.dataset import load_golden_set
from llm_eval.eval.runner import run_eval
from llm_eval.schemas import EvalReport
from llm_eval.trace import configure_logfire

logging.basicConfig(level=logging.WARNING)
console = Console()

GOLDEN_SET_PATH = Path("data/golden_set.jsonl")
EXPERIMENTS_DIR = Path("data/experiments")
FIELDS = ["title", "seniority", "work_mode", "location", "skills"]


def _print_summary(report: EvalReport) -> None:
    color = "green" if report.pass_rate >= 0.7 else "red"
    summary = Text.assemble(
        ("Model: ", "bold"), (report.model, "cyan"), "  |  ",
        ("Mean score: ", "bold"), (f"{report.mean_score:.2f}", color), "  |  ",
        ("Pass rate: ", "bold"), (f"{report.pass_rate:.0%}", color),
        f"  ({sum(e.passed for e in report.examples)}/{len(report.examples)} examples)",
    )
    console.print(Panel(summary, title="Eval Summary", border_style=color))


def _print_field_table(report: EvalReport) -> None:
    field_scores: dict[str, list[float]] = {f: [] for f in FIELDS}
    for ex in report.examples:
        seen: set[str] = set()
        for fr in ex.field_results:
            if fr.field in FIELDS and fr.field not in seen:
                field_scores[fr.field].append(fr.result.score)
                seen.add(fr.field)

    table = Table(title="Field-level mean scores", box=box.SIMPLE_HEAD)
    table.add_column("Field", style="bold")
    table.add_column("Mean score", justify="right")
    table.add_column("Pass rate", justify="right")

    for field, scores in field_scores.items():
        if not scores:
            continue
        mean = sum(scores) / len(scores)
        pr = sum(1 for s in scores if s >= 0.5) / len(scores)
        color = "green" if mean >= 0.5 else "red"
        table.add_row(field, f"[{color}]{mean:.2f}[/{color}]", f"{pr:.0%}")

    console.print(table)


def _print_example_table(report: EvalReport, category_by_id: dict[str, str]) -> None:
    table = Table(title="Per-example scores", box=box.SIMPLE_HEAD)
    table.add_column("ID", style="dim")
    table.add_column("Category")
    for f in FIELDS:
        table.add_column(f, justify="right")
    table.add_column("Overall", justify="right", style="bold")
    table.add_column("Pass")

    for ex in report.examples:
        primary: dict[str, float] = {}
        for fr in ex.field_results:
            primary.setdefault(fr.field, fr.result.score)
        field_cells = [f"{primary.get(f, 0):.2f}" for f in FIELDS]
        overall_color = "green" if ex.passed else "red"
        table.add_row(
            ex.example_id,
            category_by_id.get(ex.example_id, ""),
            *field_cells,
            f"[{overall_color}]{ex.overall_score:.2f}[/{overall_color}]",
            "[green]✓[/green]" if ex.passed else "[red]✗[/red]",
        )

    console.print(table)


def _save_artifact(report: EvalReport) -> Path:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = EXPERIMENTS_DIR / f"{report.model}_{ts}.json"
    path.write_text(report.model_dump_json(indent=2))
    return path


async def main(model: str, save: bool) -> None:
    configure_logfire()
    examples = load_golden_set(GOLDEN_SET_PATH)
    category_by_id = {ex.id: ex.category for ex in examples}
    console.print(f"Running eval on [cyan]{len(examples)}[/cyan] examples with [cyan]{model}[/cyan]…")

    report = await run_eval(examples, model=model)
    logfire.force_flush()

    _print_summary(report)
    _print_field_table(report)
    _print_example_table(report, category_by_id)

    if save:
        path = _save_artifact(report)
        console.print(f"\nArtifact saved to [cyan]{path}[/cyan]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the job-extraction eval")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to evaluate")
    parser.add_argument("--no-save", action="store_true", help="Skip saving the result artifact")
    args = parser.parse_args()
    asyncio.run(main(model=args.model, save=not args.no_save))
