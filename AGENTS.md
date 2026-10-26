# QueueMind Development Rules

QueueMind is an AI/ML queue prediction, simulation and optimization platform.

## Core stack

Prefer:

- Python
- Streamlit
- Plotly
- pandas
- NumPy
- scikit-learn
- SQLite
- OpenCV
- Ultralytics YOLO
- ByteTrack or an Ultralytics-supported tracker

Do not introduce additional frameworks unless they provide a clear benefit.

## Architecture

Keep the following layers separated:

1. configuration
2. database/storage
3. synthetic data generation
4. queue mathematics
5. forecasting
6. computer vision
7. optimization/recommendations
8. Streamlit UI

Do not build one giant `app.py`.

`app.py` should mainly compose reusable modules.

## Development behaviour

Before every significant change:

1. Inspect relevant existing code.
2. Explain briefly what you are about to change.
3. Preserve working functionality.
4. Modify the minimum necessary files.

After every change:

1. Run relevant unit tests.
2. Run the full test suite when practical.
3. Check imports.
4. Run the application when practical.
5. Inspect terminal errors.
6. Fix genuine errors before stopping.
7. Review git diff.

Never claim something works unless you actually tested it.

## Git

Use Git checkpoints.

Before starting a major milestone, verify the working tree status.

After a milestone is complete and tests pass:

- commit the milestone with a clear descriptive message

Do not include:

- `.venv`
- secrets
- passwords
- API keys
- generated temporary data
- large model files unless intentionally required

Do not force-push or rewrite Git history.

## Data integrity

Real NIT Rourkela operational data may not initially exist.

Therefore:

- synthetic data must always be labelled synthetic
- never present synthetic values as measured campus data
- never fabricate ML accuracy
- never fabricate computer-vision accuracy
- never fabricate model evaluation results

If something has not been evaluated, explicitly report that it has not been evaluated.

## Computer vision privacy

Do not implement:

- face recognition
- identity recognition
- biometric recognition
- student identification

Use anonymous temporary tracking IDs.

Prefer local frame processing.

Avoid unnecessary video retention.

## Machine learning

Start with interpretable baselines.

Always compare ML performance against a baseline.

For regression use appropriate metrics such as:

- MAE
- RMSE
- R² when meaningful

Avoid meaningless statements such as "95% accuracy" for regression.

Use chronological splits for time-dependent forecasting.

Prevent data leakage.

Do not use deep learning for forecasting unless simpler models have already been evaluated.

## Queue simulation

Queue mathematics must be deterministic and testable.

Queue length can never become negative.

Counter recommendations must come from actual simulation/optimization results rather than arbitrary hardcoded conditions.

## Configuration

Location-specific information belongs in configuration files.

Do not spread logic such as:

```python
if location == "canteen"
```

through the codebase.

The same QueueMind engine should work for all locations.

## Reliability

Professor/demo mode must remain functional even if:

- webcam is unavailable
- YOLO cannot load
- no trained forecast model exists
- no live campus data exists

Provide graceful fallbacks.

## UI

The final Streamlit interface should feel like a professional operations dashboard rather than a classroom tutorial.

Important outputs include:

- current queue
- estimated current wait
- arrival rate
- service rate
- active counters
- predicted queue at future horizons
- predicted wait
- congestion level
- recommended action
- historical queue graph
- forecast graph
- counter simulator
- annotated video when vision is available

Avoid excessive decoration.

## Documentation

Keep README updated with:

- architecture
- installation
- commands
- methodology
- synthetic-data disclosure
- privacy considerations
- model limitations
- current project status
