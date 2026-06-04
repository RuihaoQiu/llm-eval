"""Marimo notebook: side-by-side model comparison from versioned experiment artifacts.

Run with:
    uv run marimo run notebooks/model_comparison.py
Or edit mode:
    uv run marimo edit notebooks/model_comparison.py
"""

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _imports():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    import altair as alt
    import marimo as mo
    import pandas as pd

    return Path, alt, mo, pd


@app.cell
def _controls(Path, mo):
    experiments_dir = Path(__file__).parent.parent / "data" / "experiments"
    artifacts = sorted(experiments_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    options = {p.stem: p for p in artifacts}

    if len(options) < 2:
        mo.stop(True, mo.callout(mo.md("Need at least 2 experiment artifacts in `data/experiments/`."), kind="warn"))

    names = list(options.keys())
    baseline_picker = mo.ui.dropdown(options=names, value=names[0], label="Baseline")
    candidate_picker = mo.ui.dropdown(options=names, value=names[-1], label="Candidate")
    mo.hstack([baseline_picker, candidate_picker])
    return artifacts, baseline_picker, candidate_picker, experiments_dir, names, options


@app.cell
def _load_and_compare(baseline_picker, candidate_picker, mo, options, pd):
    from llm_eval.eval.experiment import compare_experiments, load_experiment

    baseline = load_experiment(options[baseline_picker.value])
    candidate = load_experiment(options[candidate_picker.value])

    rows = compare_experiments(baseline, candidate)
    df = pd.DataFrame(rows)

    mo.md(f"""
    ## {baseline_picker.value} vs {candidate_picker.value}

    **Baseline** — mean: `{baseline.mean_score:.2f}`, pass rate: `{baseline.pass_rate:.0%}`
    &nbsp;&nbsp;|&nbsp;&nbsp;
    **Candidate** — mean: `{candidate.mean_score:.2f}`, pass rate: `{candidate.pass_rate:.0%}`
    """)
    return baseline, candidate, df, rows


@app.cell
def _delta_table(df, mo):
    def _color_delta(val: float) -> str:
        if val > 0.01:
            return "color: green"
        if val < -0.01:
            return "color: red"
        return ""

    styled = (
        df.style
        .format({
            "baseline_mean": "{:.3f}",
            "candidate_mean": "{:.3f}",
            "delta_mean": "{:+.3f}",
            "baseline_pr": "{:.0%}",
            "candidate_pr": "{:.0%}",
            "delta_pr": "{:+.0%}",
        })
        .applymap(_color_delta, subset=["delta_mean", "delta_pr"])
    )
    mo.ui.table(df.rename(columns={
        "field": "Field",
        "baseline_mean": "Baseline mean",
        "candidate_mean": "Candidate mean",
        "delta_mean": "Δ mean",
        "baseline_pr": "Baseline PR",
        "candidate_pr": "Candidate PR",
        "delta_pr": "Δ PR",
    }))
    return (styled,)


@app.cell
def _bar_chart(alt, df, pd):
    # Exclude overall row for the bar chart — it lives in the summary above
    field_df = df[df["field"] != "overall"].copy()

    melted = pd.melt(
        field_df,
        id_vars=["field"],
        value_vars=["baseline_mean", "candidate_mean"],
        var_name="model",
        value_name="mean_score",
    )
    melted["model"] = melted["model"].str.replace("_mean", "")

    chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("field:N", title="Field", sort=list(field_df["field"])),
            y=alt.Y("mean_score:Q", title="Mean score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model:N", title="Model"),
            xOffset="model:N",
            tooltip=["field", "model", alt.Tooltip("mean_score:Q", format=".3f")],
        )
        .properties(title="Field-level mean scores: baseline vs candidate", width=500, height=300)
    )
    chart
    return chart, field_df, melted


@app.cell
def _delta_bar(alt, df):
    field_df2 = df[df["field"] != "overall"].copy()

    delta_chart = (
        alt.Chart(field_df2)
        .mark_bar()
        .encode(
            x=alt.X("field:N", title="Field", sort=list(field_df2["field"])),
            y=alt.Y("delta_mean:Q", title="Δ mean score"),
            color=alt.condition(
                alt.datum.delta_mean > 0,
                alt.value("#2ecc71"),
                alt.value("#e74c3c"),
            ),
            tooltip=["field", alt.Tooltip("delta_mean:Q", format="+.3f")],
        )
        .properties(title="Delta (candidate − baseline) per field", width=500, height=250)
    )
    delta_chart
    return delta_chart, field_df2
