"""Packaged Streamlit application for live run comparison."""

import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import altair as alt
import streamlit as st

from deeptool._dashboard_data import (
    DashboardRun,
    chart_records,
    discover_runs,
    metric_groups,
)


_PALETTE = (
    "#4f46e5",
    "#16a34a",
    "#dc2626",
    "#d97706",
    "#0891b2",
    "#9333ea",
    "#db2777",
    "#4d7c0f",
    "#0369a1",
    "#7c3aed",
)


def build_chart(records, metric, unit, colors):
    """Build one run-colored chart, with train/validation loss line styles."""
    data = alt.Data(values=records)
    encoding = {
        "x": alt.X("x:Q", title=unit),
        "y": alt.Y(
            "value:Q", title="Loss" if metric == "loss" else metric
        ),
        "color": alt.Color(
            "run:N",
            scale=alt.Scale(
                domain=list(colors), range=list(colors.values())
            ),
        ),
        "tooltip": ["run:N", "x:Q", "value:Q"],
    }
    if metric == "loss":
        encoding["strokeDash"] = alt.StrokeDash(
            "split:N",
            scale=alt.Scale(
                domain=["train", "validation"],
                range=[[1, 0], [6, 4]],
            ),
        )
    return alt.Chart(data).mark_line().encode(**encoding).interactive()


def render_dashboard(root: Path, refresh: float) -> None:
    """Render the live dashboard for one run root."""
    st.set_page_config(page_title="deeptool dashboard", layout="wide")
    st.title("deeptool dashboard")
    st.caption(f"Run root: {root.resolve()}")

    run_every = refresh if refresh > 0 else None
    st.sidebar.caption("Run visibility and Focus")

    @st.fragment(run_every=run_every)
    def live_panel() -> None:
        _render_live_panel(root, refresh)

    live_panel()


def _render_live_panel(root: Path, refresh: float) -> None:
    runs = discover_runs(root)
    if not runs:
        st.info(
            "No recorded runs found. Train with "
            "Trainer(log_dir='runs/exp1') and point this dashboard at runs/."
        )
        st.sidebar.caption(_refresh_text(refresh))
        return

    visibility = st.session_state.setdefault("visible_runs", {})
    colors = st.session_state.setdefault("run_colors", {})
    for run in runs:
        visibility.setdefault(run.name, True)
        if run.name not in colors:
            colors[run.name] = _PALETTE[len(colors) % len(_PALETTE)]
        visibility[run.name] = st.sidebar.toggle(
            run.name,
            value=visibility[run.name],
            key=f"run-visible::{run.name}",
        )

    visible = [run for run in runs if visibility.get(run.name, False)]
    healthy = [run for run in visible if run.error is None]
    for run in visible:
        if run.error is not None:
            st.error(f"{run.name}: {run.error}")
        if run.meta_error is not None:
            st.warning(f"{run.name}: {run.meta_error}")

    if not visible:
        st.info("Select at least one run in the sidebar.")
        st.sidebar.caption(_refresh_text(refresh))
        return
    if not healthy:
        st.info("No selected run has readable history yet.")
        st.sidebar.caption(_refresh_text(refresh))
        return

    focus_names = [run.name for run in healthy]
    focus_name = st.session_state.get("focus_run")
    if focus_name not in focus_names:
        focus_name = focus_names[0]
    focus_key = "focus-run-widget"
    if (
        focus_key in st.session_state
        and st.session_state[focus_key] not in focus_names
    ):
        st.session_state[focus_key] = focus_name
    focus_name = st.sidebar.selectbox(
        "Focus run",
        focus_names,
        index=focus_names.index(focus_name),
        key=focus_key,
    )
    st.session_state.focus_run = focus_name

    groups = metric_groups(healthy)
    selected_groups = _metric_toggles(groups)
    st.sidebar.caption(_refresh_text(refresh))
    latest_modified = max(run.modified_at for run in visible)
    st.sidebar.caption(
        "Last history change: "
        + datetime.fromtimestamp(latest_modified).astimezone().isoformat(
            timespec="seconds"
        )
    )
    st.sidebar.caption(str(root.resolve()))

    focus = next(run for run in healthy if run.name == focus_name)
    _render_focus(focus, selected_groups)
    _render_charts(healthy, selected_groups, colors)


def _metric_toggles(groups: list[str]) -> list[str]:
    selected = []
    state = st.session_state.setdefault("visible_metrics", {})
    if groups:
        st.sidebar.markdown("**Metrics**")
    for metric in groups:
        state.setdefault(metric, True)
        state[metric] = st.sidebar.toggle(
            f"Metric: {metric}",
            value=state[metric],
            key=f"metric-visible::{metric}",
        )
        if state[metric]:
            selected.append(metric)
    return selected


def _render_focus(run: DashboardRun, selected_groups: list[str]) -> None:
    st.subheader(f"Focus · {run.name}")
    custom_metrics = [metric for metric in selected_groups if metric != "loss"]
    custom_metric = next(
        (metric for metric in custom_metrics if metric in run.metrics), None
    )
    columns = st.columns(4)
    columns[0].metric(
        "Step" if run.unit == "step" else "Epoch",
        _display_value(run.progress),
    )
    columns[1].metric(
        "Train loss", _display_value(_last_value(run, "train_loss"))
    )
    columns[2].metric(
        "Validation loss", _display_value(_last_value(run, "val_loss"))
    )
    columns[3].metric(
        custom_metric or "Custom metric",
        _display_value(_last_value(run, custom_metric)),
    )

    latest, metadata = st.columns(2)
    latest.markdown("**Latest row**")
    latest.json(_latest_row(run))
    metadata.markdown("**Metadata**")
    metadata.json(run.meta)


def _render_charts(
    runs: list[DashboardRun],
    selected_groups: list[str],
    colors: dict[str, str],
) -> None:
    selected_colors = {run.name: colors[run.name] for run in runs}
    for unit, heading in (("epoch", "Epoch metrics"), ("step", "Step metrics")):
        unit_runs = [run for run in runs if run.unit == unit]
        if not unit_runs:
            continue
        st.subheader(heading)
        rendered = False
        for metric in selected_groups:
            records = chart_records(unit_runs, metric, unit)
            if not records:
                continue
            st.altair_chart(
                build_chart(records, metric, unit, selected_colors),
                width="stretch",
            )
            rendered = True
        if not rendered:
            st.info("Select a metric with recorded values.")


def _last_value(run: DashboardRun, metric: str | None):
    if metric is None:
        return None
    values = run.metrics.get(metric, [])
    return next((value for value in reversed(values) if value is not None), None)


def _latest_row(run: DashboardRun) -> dict[str, object]:
    if run.unit is None or not run.metrics.get(run.unit):
        return {}
    index = len(run.metrics[run.unit]) - 1
    return {
        metric: values[index]
        for metric, values in run.metrics.items()
        if index < len(values) and values[index] is not None
    }


def _display_value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _refresh_text(refresh: float) -> str:
    if refresh == 0:
        return "Auto-refresh off"
    return f"Refresh every {refresh:g}s"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--refresh", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the app using arguments forwarded after Streamlit's separator."""
    args = _parser().parse_args(argv)
    render_dashboard(args.runs_dir, args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
