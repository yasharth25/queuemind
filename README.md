# QueueMind ⏱️

**See your queue. Predict the wait. Take action.** [Open the academic demo](https://queuemind-nitr.streamlit.app/) · [CI](https://github.com/yasharth25/queuemind/actions/workflows/ci.yml)

QueueMind is an academic prototype for queue counting, demand forecasting, deterministic wait simulation, and counter recommendations. It is **not** a live NIT Rourkela CCTV deployment. The public app's normal mode uses clearly labelled **SYNTHETIC demo data**. It must not be interpreted as measured campus operations.

## Use the two-page app

Run `streamlit run app.py`. The two main pages are **Live Camera** and **Simulation Studio**. Live Camera offers local connector instructions, optional authenticated aggregate monitoring, the existing video uploader, and the SYNTHETIC demo. Simulation Studio freezes the latest available camera observation, or clearly uses SYNTHETIC demo data when there is no camera. Counter, crowd, extra-arrival, service-time, duration, and playback controls use the deterministic queue simulator. Simulation changes never modify observed camera counts.

**Advanced Details** keeps the previous upload, demo, predictions, counter, results, evaluation, Vision Lab, and ML Lab tools accessible. Forecast evaluation is on held-out **SYNTHETIC** data only. Real-video count accuracy appears only after manually labelled samples are supplied; otherwise it says **Not measured yet**.

### Connect a camera locally

Install optional local dependencies with `python -m pip install -r requirements-vision.txt`, then use a camera you own or are explicitly authorized to access. OpenCV and YOLO stay out of cloud `requirements.txt`.

```powershell
# Host webcam, on the computer that actually has the camera
python -m scripts.camera_connector --source webcam --webcam-index 0 --test-only

# RTSP camera, on a computer in the authorized camera network
python -m scripts.camera_connector --source rtsp --host CAMERA_IP --path STREAM_PATH --test-only

# Discover ONVIF candidates on one explicitly selected local interface
python -m scripts.camera_connector --discover --interface YOUR_LOCAL_IPV4

# Replay an authorized local video; clearly labelled REPLAY_VIDEO
python -m scripts.camera_connector --source replay_video --replay PATH_TO_VIDEO --preview
```

The connection test succeeds only after decodable frames are read. Camera credentials are prompted locally and kept in memory for the session; brand and model alone cannot authenticate or locate a stream. ONVIF discovery lists candidates and does not prove stream access. ONVIF media URI lookup uses optional `onvif-zeep` if installed; vendor-specific URLs are not guessed. Actual webcam, ONVIF, and RTSP compatibility require testing with authorized hardware. Use `--preview` to open the **local** annotated window. By default the cloud page receives aggregate counts only. With explicit `--share-preview` and a paired service, it can show the latest annotated JPEG to authenticated viewers; that frame leaves the local network and expires from service memory after 15 seconds.

For ONVIF stream URI lookup on the local connector, install the optional `onvif-zeep` package locally with `python -m pip install onvif-zeep`. Discovery itself uses a single explicit local IPv4 interface and needs no extra package.

Mark a camera-specific queue polygon and virtual entry/service lines on a preview frame:

```powershell
python -m scripts.camera_connector --source webcam --calibrate --calibration config/calibrations/my_camera.json --preview
python -m scripts.camera_connector --source webcam --calibration config/calibrations/my_camera.json --preview
```

The calibration file contains normalized geometry only and is Git-ignored. Default geometry in `config/vision.yaml` is a starting point, not a validated campus camera layout. Crossings are anonymous boundary events; a service-line crossing does **not** prove a completed transaction. A complete rolling observation window is required before a measured arrival rate is projected. Live predictions use observed entry-rate persistence plus queue simulation. The existing ML models have **not** been validated on CCTV observations.

### Local aggregate service and cloud pairing

For local end-to-end development, start the separate aggregate service in one terminal:

```powershell
python -m scripts.live_data_service --dev
```

It prints temporary admin/viewer tokens. In another terminal, set the app's trusted endpoint and run Streamlit:

```powershell
$env:QUEUEMIND_LIVE_API_ENDPOINT = 'http://127.0.0.1:8780'
python -m streamlit run app.py
```

On **Live Camera**, enter the viewer token to see paired cameras. Under **Connect Camera → Pair this workspace**, enter the admin token to create a one-use, five-minute pairing code. Run the connector on the same authorized camera network and enter that code at its local prompt:

```powershell
python -m scripts.camera_connector --source webcam --service-url http://127.0.0.1:8780 --preview
```

For remote use, configure `QUEUEMIND_LIVE_API_ENDPOINT` in private Streamlit secrets to a **trusted HTTPS** endpoint. The service itself binds to loopback only; put it behind a secured HTTPS reverse proxy on a separate host. The connector sends bounded, anonymous aggregate observations outbound by default; optional `--share-preview` sends short-lived annotated JPEG frames. The site never fetches user-entered private RTSP URLs. Viewer tokens, admin tokens, and camera credentials must not go in Git, query parameters, or logs. Pairing codes are one-use; agent credentials expire, can be revoked, and are scoped to one camera/workspace. The API supports heartbeat, latest/history, disconnect, and removal. In-memory service state is ephemeral: restart loses pairings and observations. Public multi-user operation still requires identity, durable storage, deployment, and institutional review. No such service is provisioned by this repository, so the public site remains **CONNECTOR REQUIRED** unless its operator configures one.

### Analyze a video locally

Install the optional vision dependencies, then run the app:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-vision.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open **Live Camera → Upload Video** or **Advanced Details → Uploaded Video**, upload an authorized MP4/MOV/AVI/MKV recording, choose a queue-area preset (or edit the polygon), and click **Analyze Video**. The local path option and entry/service-line calibration are under Advanced Video Settings. Local analysis accepts up to 40 MB, 60 seconds, and 1280 × 720. Quick Analysis samples frames; Full Analysis processes every frame. The default `yolo11n.pt` may download on first use; supply a preinstalled local path to work offline. Run `python -m scripts.vision_demo "C:\path\to\video.mp4"` for a standalone local test.

Person detection, anonymous IoU tracking, queue-area counting, and virtual-line crossings are real inference, not drawn examples. A bottom-centre ground point must be inside the selected polygon to count as waiting. A service-line crossing is **not** a verified completed transaction. Recalibrate the queue area and rerun if the camera view differs from the default.

After analysis, the latest frame's queue count and crossing totals are video-derived. Future arrivals require enough observed entry crossings or an **explicitly selected SYNTHETIC demand assumption**. The latter is labelled “Video count + synthetic demo demand.” The selected facility's service time is an example configuration, not a measured video service speed. The deterministic simulator then projects +5/+10/+20/+30-minute queue and wait and optimizes counters. This is not claimed to be an ML forecast from one short clip. Video state is shared across tabs; **Back to Demo** restores the independent synthetic simulation.

### Public video inference status

The Streamlit Community Cloud installation uses only `requirements.txt` and intentionally excludes OpenCV, Ultralytics, PyTorch, and model weights. Therefore **public uploaded-video analysis is not active by default**. The upload control can preview a clip, but does not fabricate people counts when no detector is configured. The SYNTHETIC demo and Simulation Studio remain usable.

An optional private video worker is included for an authorized separately hosted inference service. It reuses the local YOLO analyzer, accepts bounded video uploads, requires a bearer token, handles one analysis at a time, returns aggregate frame results and bounded annotated previews, and deletes its temporary video copy. No inference provider, credentials, or public worker is provisioned by this repository. To use it, deploy the worker on an appropriate host behind HTTPS with `requirements-vision.txt` installed and a pre-provisioned YOLO weight file, set a random `QUEUEMIND_VISION_TOKEN` (at least 24 characters), then run:

```powershell
.\.venv\Scripts\python.exe -m scripts.vision_service --host 127.0.0.1 --port 8765
```

Configure the Streamlit deployment with `QUEUEMIND_VISION_ENDPOINT=https://YOUR-AUTHORIZED-HOST/analyze` and the same `QUEUEMIND_VISION_TOKEN` via private Streamlit secrets. HTTP endpoints are accepted only for localhost testing. Do **not** commit the token. The worker enforces 40 MB, 30 seconds, 1280 × 720, at most 1800 frames, a minimum stride of 5, and a maximum of 12 annotated previews. Use a reverse proxy with TLS, request timeouts, and host-level resource/rate controls before exposing it. The public workflow must be tested with a real permitted video after configuration; no live-site video-inference claim is made yet.

## Architecture and methodology

`app.py` composes the UI; `src/ui` owns page presentation and help popovers. `src/config.py` and `config/` hold facility and vision settings. `src/synthetic` generates labelled synthetic observations; `src/storage` provides SQLite repositories, but demo startup does not require persistent storage. `src/forecasting` holds leakage-aware features and four horizon-specific `HistGradientBoostingRegressor` arrival models. A past-three-observation baseline is evaluated on the same chronologically held-out data. MAE, RMSE, and R² where meaningful are calculated, never invented. Ordinary UI changes do not retrain the ML model. `src/queueing` simulates nonnegative queue evolution and selects the smallest tested counter count meeting the configured target wait. `src/vision` handles person-only detection, anonymous tracking, normalized queue ROI, entry/service crossing events, and the optional recorded-video HTTP result contract. `src/vision/live` adds local source validation, normalized calibration, bounded monitoring, aggregate observations, measured-rate projections, immutable what-if snapshots, and a separately authenticated live-data contract.

Generated data is always **SYNTHETIC**. Queue simulation is deterministic; predicted wait is queue length divided by configured total service capacity. Video crossing rates and configured service capacity are distinct. Forecast evaluation is on synthetic data, not real NIT Rourkela operations. No real-video count accuracy is claimed without manually labelled ground truth.

## Privacy

QueueMind counts people, not identities. It has no face recognition, biometrics, student identification, age/gender inference, or identity database. Track IDs are temporary and anonymous. The local connector keeps camera credentials in memory, processes frames on the camera network, and sends only aggregate counts/events by default. `--preview` remains local; optional `--share-preview` sends one protected, short-lived annotated frame at a time to authenticated viewers. Uploaded raw videos are not saved to SQLite and temporary disk copies are removed after each local or worker analysis. Session-held uploads and bounded previews are cleared on reset or session end; processed video export is off by default. Only analyze recordings you are authorized to use. Real campus CCTV would require institutional permission and a separate privacy/deployment review.

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

The public app cannot analyze uploaded videos until a private recorded-video worker is hosted and configured; live camera monitoring separately requires a trusted HTTPS aggregate service and paired connector. Neither external service is deployed here. The public site cannot provide a remote live camera preview until a protected service is configured and a connector explicitly enables `--share-preview`. The live service uses ephemeral process memory, not durable multi-user storage or identity. Real-person footage, crowded scenes, camera calibration, local webcam/RTSP/ONVIF behavior, mobile-device upload, and public-cloud end-to-end analysis still need permitted hardware/footage verification. The tracker uses IoU, so occlusion and frame skipping can lose IDs or crossings. Short videos rarely establish a reliable future arrival rate. The ML evaluation uses only synthetic data; forecasting on real campus demand remains unvalidated.
