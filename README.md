# QueueMind — AI That Predicts Queues Before They Become Queues

QueueMind is a presentation-ready prototype for queue simulation, synthetic demand forecasting, counter optimisation, and privacy-preserving person-only queue analytics. It supports an operations dashboard and an offline Professor Demo Mode. **Synthetic data is not real NIT Rourkela operational data.**

## Problem and core idea

Queues become costly before staff can react. QueueMind combines historical/synthetic observations, future-arrival forecasting, deterministic queue mathematics, and optional camera queue counts to project congestion and recommend a minimum counter count.

## Architecture and features

Configuration → synthetic data/SQLite storage → forecasting features and regressors → queue simulation/optimisation → Streamlit UI. Optional vision is person-only YOLO detection → anonymous tracking → queue ROI → entry/service events. Features include forecast horizons (+5/+10/+20/+30), wait estimates, congestion classification, counter comparison, model evaluation, and an offline synthetic demo.

## Methodology

- **Queue and wait:** queue evolution is deterministic, never negative; wait is queued people divided by total service capacity.
- **Counter optimisation:** tests candidate capacities against the configured target wait; no hardcoded recommendations.
- **Synthetic data:** time-of-day category profiles generate physically consistent rows labelled `SYNTHETIC`.
- **ML:** chronological splits, past-only lag/rolling features, rolling-arrival baseline, and `HistGradientBoostingRegressor`; MAE, RMSE, and meaningful R² are compared on held-out synthetic data.
- **Vision/ROI/tracking:** YOLO detects only people, temporary anonymous IDs are tracked with IoU, and only bottom-centre points inside a normalized queue polygon count as queued.
- **Evaluation:** forecasting metrics are calculated on demand from held-out synthetic data. CV queue-count MAE requires manually labelled samples with `frame_id,ground_truth_queue_count,predicted_queue_count`.

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
```

Use the sidebar to open Professor Demo Mode; choose a scenario and adjust counters/demand. It remains usable without a camera, YOLO weights, a network connection, or trained model artifacts.

## Structure

`src/config.py` configuration; `src/synthetic` synthetic generator; `src/storage` SQLite repository; `src/queueing` mathematics; `src/forecasting` feature/model/integration code; `src/vision` optional anonymous vision; `src/demo` scenarios; `src/ui` Streamlit pages.

## Limitations and future work

The model is evaluated only on synthetic data. Vision is code-tested but needs local video/webcam calibration and quantitative labelled-footage evaluation. Future work includes validated ByteTrack integration, location camera configuration, real authorised data collection, and production deployment controls.
