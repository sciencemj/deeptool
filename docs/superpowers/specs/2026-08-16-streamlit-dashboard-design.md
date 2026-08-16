# Streamlit Run Dashboard Design

## Scope

Add an optional Streamlit dashboard for watching persisted deeptool runs and
comparing their metrics. The dashboard must work while training is still
appending JSONL, support safe viewing through an SSH tunnel, and keep Streamlit
and Altair out of deeptool's core dependencies.

The dashboard reads existing run artifacts. It does not control training,
modify run files, provide authentication, or implement a custom remote data
protocol. Public hosting and a separate Streamlit Cloud deployment are out of
scope.

## Chosen approach

Ship a packaged Streamlit app and a small console launcher:

```text
uv add "deeptool[dashboard]"
deeptool-dashboard runs/
```

The launcher uses Streamlit's public command-line interface through the active
Python interpreter rather than importing Streamlit's internal server APIs. The
core package remains importable without the dashboard extra. Direct Altair use
is needed for line dashes, so both Streamlit and Altair are declared only in
the `dashboard` optional extra.

This is preferable to a copied example script because it gives every project a
consistent command and upgrade path. A custom web server would duplicate
refresh, widget-state, and charting behavior that Streamlit already provides.

## Packaging and command-line API

`pyproject.toml` adds:

- a `dashboard` optional dependency group containing `streamlit>=1.37` and
  `altair>=5`;
- the `deeptool-dashboard` console entry point.

The command accepts:

```text
deeptool-dashboard RUNS_DIR
  [--host 127.0.0.1]
  [--port 8501]
  [--refresh 2]
  [--no-browser]
```

`RUNS_DIR` must be an existing directory. `--refresh` is a non-negative number
of seconds; zero disables automatic refresh. `--no-browser` maps to
Streamlit's headless mode. Host, port, headless mode, refresh interval, and the
resolved run path are forwarded as explicit arguments to the packaged app.

If the dashboard extra is absent, the launcher exits with an actionable
message containing `uv add "deeptool[dashboard]"`. It does not make core
`import deeptool` depend on Streamlit.

## Components

The implementation has three narrow parts:

1. `deeptool.dashboard` owns argument parsing, validation, and launching the
   packaged app with `python -m streamlit run`.
2. `deeptool._dashboard_data` discovers runs, performs live-safe reads, loads
   metadata, and converts arbitrary run schemas into chart-ready records. It
   has no Streamlit state and can be tested directly.
3. `deeptool._dashboard_app` owns Streamlit widgets, session state, automatic
   refresh, and Altair rendering.

The existing strict `load_runs()` contract stays unchanged. A shared private
history reader gains a live-read mode used only by the dashboard. This avoids
duplicating JSONL semantics without weakening the public loader.

## Live data flow

The Streamlit app uses `st.fragment(run_every=...)` when refresh is enabled.
On each rerun it discovers immediate child directories containing
`history.jsonl`, then reads each run independently.

A writer may be observed between writing a JSON object and its terminating
newline. Live-read mode ignores only a malformed final fragment when the file
does not end in a newline. Every preceding malformed line, and a malformed
newline-terminated final line, remains an error. The next refresh retries the
ignored fragment. Completed rows are never discarded.

Metadata is optional. When present, `meta.json` supplies model, device,
hyperparameters, start time, and `max_epochs` or `max_steps`. The dashboard
does not infer a definitive `LIVE` or `done` state from file age; it reports the
last completed progress value and file update time. This avoids declaring a
long epoch dead or an interrupted process alive.

One corrupt run produces an error card naming its file and line but does not
prevent other runs from rendering.

## Dashboard behavior

The approved layout is overview-first:

- The sidebar lists every run with an independent visibility toggle.
- A Focus run selector is limited to visible runs.
- Hiding the Focus run moves focus to the first remaining visible run.
- Hiding every run shows a selection prompt and no misleading empty charts.
- Newly discovered runs are added without resetting existing choices.
- Each run keeps a stable color for the browser session, even when new runs
  appear.
- Metric toggles are built from the union of visible-run schemas.
- KPI cards, the latest row, and metadata describe only the Focus run.

`train_loss` and `val_loss` form one `Loss` chart. Run identity is encoded by
color; train loss uses a solid line and validation loss uses a dashed line. A
run with only one loss still renders that series. All other metric names remain
uninterpreted and receive one chart per metric, overlaying visible runs that
contain it.

Epoch-based rows use `epoch` as their x axis and step-based rows use `step`.
The app never overlays those two units on one axis. If selected runs contain
both units, it renders separate Epoch and Step chart groups with the same run
visibility and metric choices.

Missing values become chart gaps. Non-numeric values are excluded from scalar
charts and identified in the run detail rather than causing chart conversion
to fail.

## Remote SSH usage and security

The safe default binds only to the training host's loopback interface:

```text
# local machine
ssh -L 8501:127.0.0.1:8501 user@training-host

# remote training host
deeptool-dashboard runs/ --no-browser
```

The user opens `http://127.0.0.1:8501` locally. This needs no dashboard
authentication because the SSH connection provides the protected transport
and the Streamlit port is not exposed publicly.

Advanced users may explicitly pass `--host 0.0.0.0 --port PORT`. Documentation
must warn that this exposes the service to reachable networks and that
deeptool does not add authentication or TLS.

## Failure behavior

- A missing or non-directory root fails before Streamlit starts.
- An empty root renders an instructional empty state.
- A missing or partial metadata document does not block history charts.
- An invalid refresh interval, host, or port produces a CLI usage error.
- A missing dashboard dependency produces the install hint.
- A corrupt completed JSONL line is isolated to its run and shown with path and
  line number.
- Streamlit subprocess failures propagate as the command's exit status.

## Verification and documentation

Tests cover:

1. Live-safe handling of a partial final fragment and strict rejection of a
   corrupt completed or middle line.
2. Independent per-run error isolation and optional metadata.
3. Arbitrary, disjoint metrics and missing values.
4. Combined loss chart records with run colors and train/validation line
   styles.
5. Separate epoch and step chart groups.
6. Visible-run state, Focus fallback, metric selection, and empty states.
7. CLI defaults and forwarding of host, port, refresh, and headless options.
8. The missing-extra install message.
9. Existing core tests without the extra, dashboard tests with the extra, and
   a built-wheel smoke test.

README and the Korean and English documentation show local launch, refresh
control, per-run toggles, the Focus run, and the safe SSH tunnel workflow.

This feature ships with the other approved additions in the next minor
deeptool release.
