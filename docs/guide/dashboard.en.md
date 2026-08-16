# Run dashboard

The dashboard reads `history.jsonl` to compare active and completed runs on one
screen. Streamlit and Altair remain optional, so the core install does not get
heavier.

## Install and run locally

```bash
uv add "deeptool[dashboard]"
deeptool-dashboard runs/
```

The safe default is `http://127.0.0.1:8501`, refreshing from disk every two
seconds. Disable automatic refresh when needed:

```bash
deeptool-dashboard runs/ --refresh 0
```

The dashboard discovers `history.jsonl` in immediate children of the run root.

```text
runs/
  exp1/history.jsonl
  exp1/meta.json
  exp2/history.jsonl
```

Because Trainer flushes every row, an active run is visible through its latest
completed point. Only an incomplete final JSONL fragment is deferred until the
next refresh. A malformed newline-terminated line is shown as an error for that
run while healthy runs keep rendering.

## Reading the screen

- Toggle each run independently in the sidebar.
- `Focus run` chooses the source for the KPI, latest-row, and `meta.json`
  panels.
- Metric toggles select which charts are visible.
- `train_loss` and `val_loss` share one loss chart. Color identifies the run;
  solid and dashed lines identify training and validation.
- Arbitrary scalars such as IoU and AP50 keep their names and receive separate
  charts.
- Epoch and step runs render in separate chart groups with separate x axes.

The displayed timestamp is the last history-file modification time, not an
inferred live/completed state. The dashboard is read-only and cannot control
training.

## Viewing training over SSH

SSH local forwarding is the safest default. On your local machine:

```bash
ssh -L 8501:127.0.0.1:8501 user@training-host
```

Then, on the remote host through that SSH session:

```bash
deeptool-dashboard runs/ --no-browser
```

Open `http://127.0.0.1:8501` locally. Streamlit stays bound to the remote
loopback interface and SSH provides the encrypted path. If local port 8501 is
already occupied, change both forwarded endpoints:

```bash
ssh -L 9000:127.0.0.1:9000 user@training-host
# remote
deeptool-dashboard runs/ --port 9000 --no-browser
```

## Direct network exposure

```bash
deeptool-dashboard runs/ --host 0.0.0.0 --port 8501 --no-browser
```

!!! danger "No authentication or TLS"
    deeptool does not add authentication or TLS. Binding to `0.0.0.0` lets
    anyone who can reach that port see run paths, metadata, and metrics. Use an
    SSH tunnel unless you have configured your own firewall and trusted private
    network.

## Next

- [Trainer](trainer.md) — JSONL recording and step/epoch training
- [Best weights & early stopping](best.md) — monitors to compare with Focus
