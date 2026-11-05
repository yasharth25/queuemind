# QueueMind ⏱️

**Know your queue before it gets long.** [Open the academic demo](https://queuemind-nitr.streamlit.app/) · [CI](https://github.com/yasharth25/queuemind/actions/workflows/ci.yml)

QueueMind is an academic prototype for queue counting, demand forecasting, deterministic wait simulation, and counter recommendations. It is **not** a live NIT Rourkela CCTV deployment. The public app's normal mode uses clearly labelled **SYNTHETIC demo data**. It must not be interpreted as measured campus operations.

## Use the website

Run `streamlit run app.py`. The six tabs are **Home**, **Upload Video**, **Live Demo**, **Predictions**, **Best Counters**, and **Results**. Home has the same video uploader and analysis state as Upload Video. Live Demo runs without a camera, weights, or trained forecast model. Change the crowd, facility, counters, or simulation speed to see deterministic changes. Predictions and Best Counters use simulator outputs, not static example values.

The Results tab shows actual held-out **synthetic** forecast evaluation only after the user requests it. Advanced ML and vision diagnostics remain available there. Real-video count accuracy is shown only after manually labelled samples are supplied; otherwise it says **Not measured yet**.

### Analyze a video locally

Install the optional vision dependencies, then run the app:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-vision.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open **Home** or **Upload Video**, upload an authorized MP4/MOV/AVI/MKV recording, choose a queue-area preset (or edit the polygon), and click **Analyze Video**. The local path option and entry/service-line calibration are under Advanced Video Settings. Local analysis accepts up to 40 MB, 60 seconds, and 1280 × 720. Quick Analysis samples frames; Full Analysis processes every frame. The default `yolo11n.pt` may download on first use; supply a preinstalled local path to work offline. Run `python -m scripts.vision_demo "C:\path\to\video.mp4"` for a standalone local test.

Person detection, anonymous IoU tracking, queue-area counting, and virtual-line crossings are real inference, not drawn examples. A bottom-centre ground point must be inside the selected polygon to count as waiting. A service-line crossing is **not** a verified completed transaction. Recalibrate the queue area and rerun if the camera view differs from the default.

After analysis, the latest frame's queue count and crossing totals are video-derived. Future arrivals require enough observed entry crossings or an **explicitly selected SYNTHETIC demand assumption**. The latter is labelled “Video count + synthetic demo demand.” The selected facility's service time is an example configuration, not a measured video service speed. The deterministic simulator then projects +5/+10/+20/+30-minute queue and wait and optimizes counters. This is not claimed to be an ML forecast from one short clip. Video state is shared across tabs; **Back to Demo** restores the independent synthetic simulation.

### Public video inference status

The Streamlit Community Cloud installation uses only `requirements.txt` and intentionally excludes OpenCV, Ultralytics, PyTorch, and model weights. Therefore **public uploaded-video analysis is not active by default**. The upload control can preview a clip, but does not fabricate people counts when no detector is configured. The Live Demo remains fully usable.

An optional private video worker is included for an authorized separately hosted inference service. It reuses the local YOLO analyzer, accepts bounded video uploads, requires a bearer token, handles one analysis at a time, returns aggregate frame results and bounded annotated previews, and deletes its temporary video copy. No inference provider, credentials, or public worker is provisioned by this repository. To use it, deploy the worker on an appropriate host behind HTTPS with `requirements-vision.txt` installed and a pre-provisioned YOLO weight file, set a random `QUEUEMIND_VISION_TOKEN` (at least 24 characters), then run:

```powershell
.\.venv\Scripts\python.exe -m scripts.vision_service --host 127.0.0.1 --port 8765
```

Configure the Streamlit deployment with `QUEUEMIND_VISION_ENDPOINT=https://YOUR-AUTHORIZED-HOST/analyze` and the same `QUEUEMIND_VISION_TOKEN` via private Streamlit secrets. HTTP endpoints are accepted only for localhost testing. Do **not** commit the token. The worker enforces 40 MB, 30 seconds, 1280 × 720, at most 1800 frames, a minimum stride of 5, and a maximum of 12 annotated previews. Use a reverse proxy with TLS, request timeouts, and host-level resource/rate controls before exposing it. The public workflow must be tested with a real permitted video after configuration; no live-site video-inference claim is made yet.

## Architecture and methodology

`app.py` composes the UI; `src/ui` owns page presentation and help popovers. `src/config.py` and `config/` hold facility and vision settings. `src/synthetic` generates labelled synthetic observations; `src/storage` provides SQLite repositories, but demo startup does not require persistent storage. `src/forecasting` holds leakage-aware features and four horizon-specific `HistGradientBoostingRegressor` arrival models. A past-three-observation baseline is evaluated on the same chronologically held-out data. MAE, RMSE, and R² where meaningful are calculated, never invented. Ordinary UI changes do not retrain the ML model. `src/queueing` simulates nonnegative queue evolution and selects the smallest tested counter count meeting the configured target wait. `src/vision` handles person-only detection, anonymous tracking, normalized queue ROI, entry/service crossing events, and the optional HTTP result contract.

Generated data is always **SYNTHETIC**. Queue simulation is deterministic; predicted wait is queue length divided by configured total service capacity. Video crossing rates and configured service capacity are distinct. Forecast evaluation is on synthetic data, not real NIT Rourkela operations. No real-video count accuracy is claimed without manually labelled ground truth.

## Privacy

QueueMind counts people, not identities. It has no face recognition, biometrics, student identification, age/gender inference, or identity database. Track IDs are temporary and anonymous. Uploaded raw videos are not saved to SQLite and temporary disk copies are removed after each local or worker analysis. Session-held uploads and bounded previews are cleared on reset or session end; processed video export is off by default. Only analyze recordings you are authorized to use. Real campus CCTV would require institutional permission and a separate privacy/deployment review.

## Installation, tests, and evaluation

For cloud-safe simulation and tests:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Additional commands: `python -m scripts.generate_synthetic_data --days 60`, `python -m scripts.initialize_database`, and `python -m scripts.evaluate_vision --labels "C:\path\to\vision_labels.csv"`. Do not commit generated datasets, test videos, model weights, `.env`, or secrets.

## Current limitations

The public app cannot analyze videos until a private worker is hosted and configured; that external setup has not been performed. Real-person footage, crowded scenes, camera calibration, local webcam behavior, mobile-device upload, and public-cloud end-to-end analysis still need permitted hardware/footage verification. The tracker uses IoU, so occlusion and frame skipping can lose IDs or crossings. Short videos rarely establish a reliable future arrival rate. The ML evaluation uses only synthetic data; forecasting on real campus demand remains unvalidated.
