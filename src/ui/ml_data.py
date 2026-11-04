"""Exploratory views over explicitly synthetic evaluation observations."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.forecasting.features import FEATURE_COLUMNS, build_features
from src.forecasting.forecaster import DEFAULT_HORIZONS


DATA_DICTIONARY = {
    "timestamp": "Synthetic five-minute observation time.",
    "location_id": "Configured facility identifier.",
    "category": "Configured facility category.",
    "active_counters": "Number of simulated serving counters.",
    "arrivals": "Simulated arrivals within an interval.",
    "departures": "Simulated completed services within an interval.",
    "queue_count": "Simulated waiting people at interval end.",
    "average_service_time": "Configured mean minutes per service per counter.",
    "arrival_rate": "Synthetic demand rate in people per minute.",
    "service_rate": "Counter service capacity in people per minute.",
    "data_source": "SYNTHETIC source disclosure marker.",
}

FEATURE_DEFINITIONS = {
    "hour": "Hour (0–23) of the prediction timestamp.",
    "minute": "Minute (0–59) of the prediction timestamp.",
    "day_of_week": "Weekday of the prediction timestamp; Monday = 0.",
    "hour_sin": "Sine encoding of current hour on a 24-hour cycle.",
    "hour_cos": "Cosine encoding of current hour on a 24-hour cycle.",
    "day_sin": "Sine encoding of current weekday on a seven-day cycle.",
    "day_cos": "Cosine encoding of current weekday on a seven-day cycle.",
    "queue_count": "People waiting in the current observation.",
    "active_counters": "Active serving counters in the current observation.",
    "arrival_rate": "Current synthetic arrival-rate assumption (people/min).",
    "service_rate": "Current total service capacity (people/min).",
    "arrivals_lag_1": "Arrivals in the immediately preceding observation.",
    "queue_lag_1": "Queue count in the immediately preceding observation.",
    "arrivals_rolling_mean_3": "Mean arrivals in up to three observations strictly before now.",
    "queue_rolling_mean_3": "Mean queue count in up to three observations strictly before now.",
}


def render_data(locations, data):
    st.markdown("### Data explorer · SYNTHETIC DATA")
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp)
    frame["day_name"] = frame.timestamp.dt.day_name()
    frame["wait_estimate_minutes"] = frame.queue_count / frame.service_rate
    names = {item["id"]: item["display_name"] for item in locations}
    frame["facility"] = frame.location_id.map(names)
    frame["congestion"] = "Normal"
    for item in locations:
        scope = frame.location_id.eq(item["id"])
        frame.loc[scope & frame.queue_count.ge(item["warning_queue_length"]), "congestion"] = "Warning"
        frame.loc[scope & frame.queue_count.ge(item["critical_queue_length"]), "congestion"] = "Critical"
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(frame):,}")
    b.metric("Date span", f"{frame.timestamp.dt.date.nunique()} days")
    c.metric("Facilities", frame.location_id.nunique())
    d.metric("Missing cells", int(frame[list(DATA_DICTIONARY)].isna().sum().sum()))
    st.caption(f"SYNTHETIC · {frame.timestamp.min():%d %b %Y %H:%M} to {frame.timestamp.max():%d %b %Y %H:%M} · five-minute cadence · seed 9")
    a, b, c, d = st.columns(4)
    facilities = a.multiselect("Facility", sorted(frame.facility.unique()), default=sorted(frame.facility.unique()))
    weekdays = list(pd.date_range("2026-01-05", periods=7).day_name())
    days = b.multiselect("Day of week", weekdays, default=weekdays)
    risks = c.multiselect("Congestion", ["Normal", "Warning", "Critical"], default=["Normal", "Warning", "Critical"])
    hours = d.slider("Hour of day", 0, 23, (0, 23))
    maximum = int(frame.queue_count.max())
    low, high = st.slider("Queue level", 0, maximum, (0, maximum))
    filtered = frame[frame.facility.isin(facilities) & frame.day_name.isin(days)
                     & frame.congestion.isin(risks) & frame.timestamp.dt.hour.between(*hours)
                     & frame.queue_count.between(low, high)]
    st.caption(f"{len(filtered):,} SYNTHETIC rows match the filters.")
    if filtered.empty:
        st.info("No synthetic observations match these filters.")
        return
    focus_id = st.selectbox("Chart facility", sorted(filtered.facility.unique()))
    focus = filtered[filtered.facility.eq(focus_id)]
    focus = focus.iloc[::max(1, len(focus)//1000)]
    chart = go.Figure()
    for column, color in (("arrival_rate", "#F36F5D"), ("queue_count", "#173B63"), ("wait_estimate_minutes", "#6B4EAA")):
        chart.add_trace(go.Scatter(x=focus.timestamp, y=focus[column], name=column.replace("_", " ").title(), line_color=color))
    chart.update_layout(height=330, title="SYNTHETIC arrivals, queue and wait over time")
    st.plotly_chart(chart, width="stretch")
    left, right = st.columns(2)
    left.plotly_chart(px.bar(filtered.groupby(filtered.timestamp.dt.hour).arrival_rate.mean().reset_index(name="rate"), x="timestamp", y="rate", title="Hour-of-day demand"), width="stretch")
    right.plotly_chart(px.bar(filtered.groupby("day_name").queue_count.mean().reset_index(), x="day_name", y="queue_count", title="Day-of-week queue"), width="stretch")
    st.markdown("#### Facility and category comparison")
    st.dataframe(filtered.groupby(["facility", "category"]).agg(rows=("queue_count", "size"), mean_queue=("queue_count", "mean"), mean_arrival_rate=("arrival_rate", "mean")).reset_index(), hide_index=True, width="stretch")
    st.markdown("#### Rows by day and category")
    st.dataframe(filtered.groupby([filtered.timestamp.dt.date.rename("date"), "category"]).size().reset_index(name="rows"), hide_index=True, width="stretch")
    st.markdown("#### Future-arrivals target statistics")
    targets = build_features(data)[[f"target_arrivals_{h}m" for h in DEFAULT_HORIZONS]]
    st.dataframe(targets.describe().T, width="stretch")
    st.markdown("#### Synthetic observation preview")
    st.dataframe(filtered[list(DATA_DICTIONARY)].head(150), hide_index=True, width="stretch")
    st.download_button("DOWNLOAD FILTERED SYNTHETIC DATA", filtered[list(DATA_DICTIONARY)].to_csv(index=False), "queuemind_synthetic_filtered.csv", "text/csv")
    st.markdown("#### Data dictionary")
    st.dataframe(pd.DataFrame([{"Field": name, "Definition": definition} for name, definition in DATA_DICTIONARY.items() if name in data.columns]), hide_index=True, width="stretch")


def feature_catalog():
    """Describe the exact feature list imported from the forecasting pipeline."""
    rows = []
    for name in FEATURE_COLUMNS:
        if name.endswith("lag_1"):
            kind, lookback = "Lag", "Previous observation (5 min)"
        elif "rolling_mean_3" in name:
            kind, lookback = "Rolling", "Previous up to 3 observations (15 min)"
        elif name in {"hour", "minute", "day_of_week", "hour_sin", "hour_cos", "day_sin", "day_cos"}:
            kind, lookback = "Time", "Current timestamp"
        else:
            kind, lookback = "System", "Current observation"
        rows.append({"Feature": name, "Type": kind, "Definition": FEATURE_DEFINITIONS[name],
                     "Lookback": lookback, "Past/present only": True})
    return pd.DataFrame(rows)


def render_features(locations, data):
    st.markdown("### Feature engineering · actual pipeline")
    st.caption("This table reads FEATURE_COLUMNS from the code; target columns are not model inputs.")
    st.dataframe(feature_catalog(), hide_index=True, width="stretch")
    selected = st.selectbox("Feature snapshot · facility", sorted(data.location_id.unique()),
                            format_func=lambda key: next((item["display_name"] for item in locations if item["id"] == key), key))
    current = build_features(data[data.location_id.eq(selected)]).iloc[-1]
    st.caption(f"Latest SYNTHETIC historical input at {current.timestamp:%d %b %Y %H:%M}; not the live simulation state.")
    st.dataframe(pd.DataFrame({"Feature": FEATURE_COLUMNS, "Value": [str(current[key]) for key in FEATURE_COLUMNS]}), hide_index=True, width="stretch")
    target = st.selectbox("Correlation target", [f"target_arrivals_{h}m" for h in DEFAULT_HORIZONS], index=2)
    subset = build_features(data)[FEATURE_COLUMNS + [target]].dropna()
    st.plotly_chart(px.imshow(subset.corr(numeric_only=True), color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto", title="SYNTHETIC feature correlation"), width="stretch")
    st.caption("Exploratory correlation does not imply causation. Only past/current values feed the model.")
