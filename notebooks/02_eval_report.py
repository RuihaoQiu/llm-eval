"""Marimo notebook: visualize eval results.

Run with:
    uv run marimo run notebooks/02_eval_report.py
Or edit mode:
    uv run marimo edit notebooks/02_eval_report.py
"""

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _imports():
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    import altair as alt
    import marimo as mo
    import pandas as pd

    return Path, alt, mo, pd


@app.cell
def _controls(mo):
    model_input = mo.ui.dropdown(
        options=["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"],
        value="gpt-4o-mini",
        label="Model",
    )
    run_button = mo.ui.run_button(label="Run eval")
    mo.hstack([model_input, run_button])
    return model_input, run_button


@app.cell
async def _run_eval(Path, mo, model_input, run_button):
    from llm_eval.eval.dataset import load_golden_set
    from llm_eval.eval.runner import run_eval
    from llm_eval.schemas import EvalReport

    mo.stop(not run_button.value, mo.callout(mo.md("Press **Run eval** to start."), kind="info"))

    golden_path = Path(__file__).parent.parent / "data" / "golden_set.jsonl"
    examples = load_golden_set(golden_path)

    with mo.status.spinner(title=f"Running eval on {len(examples)} examples…"):
        report: EvalReport = await run_eval(examples, model=model_input.value)

    mo.callout(
        mo.md(
            f"**Model:** `{report.model}` &nbsp;|&nbsp; "
            f"**Mean score:** {report.mean_score:.2f} &nbsp;|&nbsp; "
            f"**Pass rate:** {report.pass_rate:.0%} ({sum(e.passed for e in report.examples)}/{len(report.examples)})"
        ),
        kind="success" if report.pass_rate >= 0.7 else "warn",
    )
    return (report,)


@app.cell
def _per_example_table(mo, pd, report: "EvalReport"):
    rows = []
    for ex in report.examples:
        row = {"id": ex.example_id, "overall": round(ex.overall_score, 2), "passed": ex.passed}
        for fr in ex.field_results:
            # keep primary score per field (first seen)
            if fr.field not in row:
                row[fr.field] = round(fr.result.score, 2)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("id")
    col_order = ["overall", "passed", "title", "seniority", "work_mode", "location", "skills"]
    df = df[[c for c in col_order if c in df.columns]]

    mo.vstack([
        mo.md("### Per-example scores"),
        mo.ui.table(df.reset_index(), selection=None),
    ])
    return (rows,)


@app.cell
def _field_bar_chart(alt, mo, pd, rows):
    field_cols = ["title", "seniority", "work_mode", "location", "skills"]
    field_rows = []
    for r in rows:
        for field in field_cols:
            if field in r:
                field_rows.append({"field": field, "score": r[field], "example": r["id"]})

    fdf = pd.DataFrame(field_rows)
    mean_df = fdf.groupby("field", as_index=False)["score"].mean().rename(columns={"score": "mean_score"})

    bar = (
        alt.Chart(mean_df)
        .mark_bar()
        .encode(
            x=alt.X("field:N", sort=field_cols, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("mean_score:Q", scale=alt.Scale(domain=[0, 1]), title="Mean score"),
            color=alt.condition(
                alt.datum.mean_score >= 0.5,
                alt.value("#4caf50"),
                alt.value("#f44336"),
            ),
            tooltip=["field", alt.Tooltip("mean_score:Q", format=".2f")],
        )
        .properties(title="Mean score per field", width=500, height=300)
    )

    threshold = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")

    mo.vstack([
        mo.md("### Field-level accuracy"),
        mo.ui.altair_chart(bar + threshold),
    ])
    return (fdf,)


@app.cell
def _score_distribution(alt, fdf, mo):
    box = (
        alt.Chart(fdf)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("field:N", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("field:N", legend=None),
        )
        .properties(title="Score distribution per field", width=500, height=300)
    )

    mo.vstack([
        mo.md("### Score distribution"),
        mo.ui.altair_chart(box),
    ])
    return


@app.cell
def _failures(mo, pd, report: "EvalReport"):
    failures = []
    for _ex in report.examples:
        if not _ex.passed:
            for _fr in _ex.field_results:
                if not _fr.result.passed:
                    failures.append({
                        "example": _ex.example_id,
                        "field": _fr.field,
                        "scorer": _fr.scorer,
                        "score": round(_fr.result.score, 2),
                        "reason": _fr.result.reason or "",
                    })

    if not failures:
        mo.callout(mo.md("No failures — all examples passed."), kind="success")
    else:
        mo.vstack([
            mo.md(f"### Failures ({len(failures)} field failures across failed examples)"),
            mo.ui.table(pd.DataFrame(failures), selection=None),
        ])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
