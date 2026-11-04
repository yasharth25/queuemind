# QueueMind ⏱️

**See now. Predict next. Act early.**

[![Open Live App](https://img.shields.io/badge/OPEN%20LIVE%20APP-QueueMind-F97360?style=for-the-badge)](https://queuemind-nitr.streamlit.app/) [![QueueMind CI](https://github.com/yasharth25/queuemind/actions/workflows/ci.yml/badge.svg)](https://github.com/yasharth25/queuemind/actions/workflows/ci.yml)

Interactive AI/ML queue forecasting, simulation and counter-optimization prototype.

**Live demo:** https://queuemind-nitr.streamlit.app/  
The live demo tracks the stable `main` branch, so tested pushes are reflected in the deployed application.

QueueMind is an academic prototype for queue simulation, synthetic demand forecasting, counter optimisation, and privacy-preserving person-only queue analytics. Its seven-tab Streamlit workspace is a professor demonstration. **Synthetic data is not real NIT Rourkela operational data.**

## Streamlit Community Cloud academic demo

Deploy `app.py` with `requirements.txt`. The cloud app is intentionally a **SYNTHETIC / SIMULATION** academic demonstration: it needs no webcam, YOLO weights, real campus data, SQLite persistence, or trained-model artifact. Person-only video inference remains an optional local capability and is not installed by the cloud requirements; Vision Lab still shows its architecture, privacy notes, and manual-label evaluation without OpenCV or Ultralytics.

## Problem and core idea

Queues become costly before staff can react. QueueMind combines historical/synthetic observations, future-arrival forecasting, deterministic queue mathematics, and optional camera queue counts to project congestion and recommend a minimum counter count.

## Architecture and features

Configuration → synthetic data/SQLite storage → forecasting features and regressors → queue simulation/optimisation → Streamlit UI. Optional vision is person-only YOLO detection → anonymous tracking → queue ROI → entry/service events. Features include forecast horizons (+5/+10/+20/+30), wait estimates, congestion classification, counter comparison, model evaluation, and an offline synthetic demo.

## Methodology

- **Queue and wait:** queue evolution is deterministic, never negative; wait is queued people divided by total service capacity.
- **Counter optimisation:** tests candidate capacities against the configured target wait; no hardcoded recommendations.
- **Synthetic data:** time-of-day category profiles generate physically consistent rows labelled `SYNTHETIC`.
- **ML:** past-only lag/rolling features and a global, non-overlapping chronological train/validation/test split. Final fitting combines train and validation; rows whose future target crosses into test are excluded. Four `HistGradientBoostingRegressor` models predict future five-minute arrival counts. A past-three-arrivals rolling baseline is evaluated on the identical held-out rows. MAE, RMSE, and meaningful R² are shown separately per horizon.
- **Vision/ROI/tracking:** local YOLO (`yolo11n.pt` by default) detects only people, temporary anonymous IDs are tracked with IoU, and only bottom-centre points inside a calibrated normalized queue polygon count as queued. Finite oriented entry/service lines count negative-to-positive track crossings with a cooldown. These are crossings, not verified service completions.
- **Vision-to-decision:** a selected video frame supplies its observed queue and trailing entry/service crossing rates. The observed entry rate is held constant as an explainable baseline, then the existing deterministic queue simulator and counter optimizer produce +5/+10/+20/+30 minute projections. Short clips are **not** presented as ML arrival forecasts; configured counter service capacity is kept distinct from the observed service-crossing rate.
- **Evaluation:** forecasting metrics are calculated only when requested in ML Lab. Held-out residuals, permutation importance on a held-out subset, and empirical absolute-error bands are available. These bands are **not** formal calibrated confidence intervals. Vision queue-count metrics require manually labelled samples with `frame_id,ground_truth_queue_count,predicted_queue_count`; no vision accuracy is claimed without such labels.

## Privacy and responsible AI

No face recognition, biometric recognition, student identification, or identity database is implemented. Track IDs are temporary and anonymous. Raw/processed video retention defaults to false in `config/privacy.yaml`. Real campus CCTV use requires institutional authorization, calibration, and applicable privacy policy; aggregate statistics are preferable to raw-video storage.

## Installation and commands

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.generate_synthetic_data --days 60
.\.venv\Scripts\python.exe -m scripts.vision_demo "C:\path\to\video.mp4"
.\.venv\Scripts\python.exe -m scripts.vision_demo 0
.\.venv\Scripts\python.exe -m scripts.evaluate_vision --labels "C:\path\to\vision_labels.csv"
.\.venv\Scripts\python.exe -m scripts.evaluate_vision --labels "C:\path\to\vision_labels.csv" --output-csv "C:\path\to\per_frame_errors.csv"
```

For a local vision environment, install `requirements-vision.txt` instead of the cloud-focused core requirements. Then launch with `python -m streamlit run app.py` (or the `.venv` command above), open **Vision Lab → Video Analyzer**, upload an MP4/AVI/MOV/MKV or enter a local file path, and inspect the decoded first-frame preview and metadata. Browser video playback depends on the codec; OpenCV can still analyze a file whose browser preview cannot play. Configure the queue polygon and oriented ENTRY/SERVICE lines in **ROI Setup** and use **TEST ROI** before running Quick (stride 1/2/3/5/10) or Full analysis. The default lightweight `yolo11n.pt` weights may download on first *local* use if absent; choose a local model path if offline. No weights are loaded on the public cloud page.

After analysis, **Live Metrics** shows bounded annotated frame previews, counts, crossing rates, track diagnostics, and a frame table; **Timeline** shows queue/event and entry/service-rate charts. **Forecast Integration** selects a frame or the latest state, a facility profile, and active counters; **USE THIS VISION STATE** makes Overview, Forecast, and Optimizer use the observed state with a prominent local-video source badge. **RETURN TO SYNTHETIC DEMO** restores the untouched simulation. **Evaluation** exports frame metrics, anonymous events, and evenly sampled predictions with blank ground-truth cells for manual labelling; upload completed labels to calculate queue-count error. Optional annotated MP4 output is off by default. Reset clears the analysis and the session-held upload. Processing is streaming: all frame statistics are retained in the session, but only a configurable bounded set of annotated previews is kept.

Open the app; it starts in Overview with a non-zero, deterministic **SYNTHETIC Canteen Lunch Rush** snapshot. Seven URL-linked tabs are available: Overview, Live Simulation, Forecast, Optimizer, ML Lab, Vision Lab, and Evaluation. The scenario controls adjust counters, demand, and accelerated time; Play, Pause, Reset, and Surge preserve state while switching tabs. Overview shows the queue, action, timeline, alerts, counter what-if, heatmap, and ML/Vision snapshots. Forecast and Optimizer use deterministic queue mathematics, not a hidden ML model. ML Lab has eleven selectable lifecycle sections: synthetic data explorer/download, feature catalog, training split, model settings, held-out predictions/metrics, residuals, held-out permutation importance, empirical error bands, and integrity. Run its evaluation explicitly; it is cached, not retrained every second. Vision Lab has seven internal sections for local analysis, ROI setup, metrics, timelines, forecast integration, manual evaluation, and privacy. Evaluation summarizes only evidence actually calculated in this session.

The simulation works without a camera, YOLO weights, network connection, live campus data, or persistent model artifact. Local YOLO/video processing is separate from the cloud deployment. Cloud camera access is disabled. An uploaded video is held only in the local Streamlit browser session for calibration/reruns until Reset or session end; each temporary disk copy is deleted after inspection/analysis. It is not stored in SQLite or committed. An uploaded labelled CSV is not saved by the app, and its provenance is not automatically verified.

## Structure

`app.py` composes `src/ui/product_shell.py` (seven-tab navigation), `operations_tabs.py` (operations/evidence), `vision_lab.py` (local-video UI), and the lazy `ml_lab.py`, `ml_data.py`, and `ml_diagnostics.py` views. `src/config.py` holds configuration; `src/synthetic` generates synthetic observations; `src/storage` is optional SQLite storage; `src/queueing` contains deterministic math; `src/forecasting` contains feature/model/integration code; `src/evaluation` contains pure metrics; `src/vision/video_io.py` and `analyzer.py` handle optional streaming video, with `config/vision.yaml` as a calibration starting point; `src/demo` holds scenarios/live state. `command_center.py` retains reusable presentation helpers and a legacy renderer, not the active entrypoint.

## Limitations and future work

The model is evaluated only on synthetic data. Current ML targets are future **interval** arrival counts, not cumulative horizon demand. The visible live forecast is a deterministic scenario projection; ML Lab inference on the latest synthetic observation is separately labelled. Empirical bands use global held-out error quantiles and are not calibrated probabilities. Vision uses an IoU anonymous-tracking baseline, not ByteTrack; it can lose IDs under occlusion, and frame skipping may miss brief crossings. The local video workflow was smoke-tested with a disposable synthetic MP4, but real person footage, calibrated ROI/lines, webcam operation, and manually labelled queue-count accuracy still require user verification. Future work includes validated ByteTrack integration, location camera configuration, authorised real data collection, drift assessment, and production deployment controls.
