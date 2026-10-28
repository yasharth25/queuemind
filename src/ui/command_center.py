"""Single-page, synthetic QueueMind Command Center."""

from __future__ import annotations

import importlib.util

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.demo.live import (
    SPEEDS, advance, initial_state, pause, play, reset, set_counters,
    trigger_surge,
)
from src.demo.scenarios import SCENARIOS
from src.forecasting.forecaster import ArrivalForecaster
from src.queueing.simulator import evaluate_counter_counts
from src.synthetic.generator import generate_synthetic_queue_data
from src.ui.command_model import (
    HORIZONS, TIME_SLOTS, WEEKDAYS, alert_items, heatmap_data,
    ml_improvement, risk_for, snapshot,
)
from src.ui.command_views import (
    action_center, alert_feed, event_stream, hero, kpi_strip, live_queue,
    section, timeline, top_nav,
)

SCENARIO_OPTIONS = (
    ("Bank Rush", "BANK RUSH"),
    ("Canteen Lunch Rush", "CANTEEN LUNCH"),
    ("Library Evening", "LIBRARY EVENING"),
    ("Health Centre Sudden Surge", "HEALTH SURGE"),
)


def _sync_controls():
    state = st.session_state.qm_live
    _rotate_counter_widget()
    st.session_state.qm_demand = float(state["demand_intensity"])
    st.session_state.qm_speed = state["simulation_speed"]


def _rotate_counter_widget():
    version = st.session_state.get("qm_counter_version", 0) + 1
    st.session_state.qm_counter_version = version
    st.session_state[f"qm_counters_{version}"] = st.session_state.qm_live["active_counters"]


def _select_scenario(name):
    st.session_state.qm_live = initial_state(name)
    st.session_state.pop("qm_comparison_from", None)
    _sync_controls()


def _change_counters():
    widget_key = f'qm_counters_{st.session_state.qm_counter_version}'
    st.session_state.qm_comparison_from = st.session_state.qm_live["active_counters"]
    set_counters(st.session_state.qm_live, st.session_state[widget_key])


def _change_demand():
    state = st.session_state.qm_live
    state["demand_intensity"] = st.session_state.qm_demand
    state["current_arrival_rate"] = (
        SCENARIOS[state["selected_scenario"]]["arrival_rate"]
        * state["demand_intensity"]
    )


def _change_speed():
    st.session_state.qm_live["simulation_speed"] = st.session_state.qm_speed


def _reset():
    st.session_state.qm_live = reset(st.session_state.qm_live)
    st.session_state.pop("qm_comparison_from", None)
    _sync_controls()


def _surge():
    trigger_surge(st.session_state.qm_live)
    _sync_controls()


@st.cache_data(show_spinner=False)
def _history(location, days, seed):
    return generate_synthetic_queue_data([location], "2026-01-05", days=days, seed=seed)


@st.cache_data(show_spinner=False)
def _evaluation(locations):
    observations = generate_synthetic_queue_data(
        list(locations), "2026-01-05", days=8, seed=9,
    )
    return ArrivalForecaster().train(observations)


def _forecast_chart(state, view, history):
    metric = st.radio(
        "Forecast metric", ("QUEUE", "WAIT TIME", "ARRIVAL RATE"),
        horizontal=True, label_visibility="collapsed", key="qm_chart_metric",
    )
    config = view["config"]
    prior = history.tail(12).copy()
    live_history = state.get("queue_history", [])
    use_live_queue = metric == "QUEUE" and len(live_history) > 1
    x_past = list(range(-55, 5, 5))
    if metric == "QUEUE":
        shift = config["queue"] - float(prior["queue_count"].iloc[-1])
        y_past = (prior["queue_count"] + shift).clip(lower=0).tolist()
        y_future = [point["queue"] for point in view["timeline"]]
        if use_live_queue:
            recent = live_history[-30:]
            elapsed = state["simulation_elapsed_minutes"]
            x_past = [point["minute"] - elapsed for point in recent]
            y_past = [point["queue"] for point in recent]
        threshold = config["warning"]
        y_label = "People waiting"
    elif metric == "WAIT TIME":
        shift = config["queue"] - float(prior["queue_count"].iloc[-1])
        adjusted = (prior["queue_count"] + shift).clip(lower=0)
        y_past = (adjusted / view["service_rate"]).tolist()
        y_future = [point["wait"] for point in view["timeline"]]
        threshold = config["target_wait"]
        y_label = "Minutes"
    else:
        y_past = prior["arrival_rate"].tolist()
        y_future = [view["arrival_rate"]] * 5
        threshold = view["service_rate"]
        y_label = "People / min"
    # The prehistory is deterministic, synthetic, and shifted to connect to
    # the scenario snapshot; it is context, not measured campus observations.
    y_past[-1] = y_future[0]
    figure = go.Figure()
    figure.add_vrect(x0=0, x1=30, fillcolor="#FFF0EB", opacity=.6, line_width=0)
    figure.add_hline(
        y=threshold, line_color="#E5A128", line_dash="dot",
        annotation_text="Target / capacity", annotation_position="top left",
    )
    figure.add_trace(go.Scatter(
        x=x_past, y=y_past, mode="lines",
        name="Simulated live history" if use_live_queue else "Synthetic context",
        line={"color": "#173B63", "width": 3},
        hovertemplate="%{x} min · %{y:.1f}<extra>History</extra>",
    ))
    figure.add_trace(go.Scatter(
        x=[point["minute"] for point in view["timeline"]],
        y=y_future, mode="lines+markers", name="Simulated future",
        line={"color": "#F36F5D", "width": 3, "dash": "dash"},
        marker={"size": 8},
        hovertemplate="+%{x} min · %{y:.1f}<extra>Forecast</extra>",
    ))
    figure.add_vline(x=0, line_color="#718096", line_width=1)
    figure.update_layout(
        height=270, margin={"l": 10, "r": 15, "t": 8, "b": 5},
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font={"family": "Manrope, sans-serif", "color": "#52677D", "size": 11},
        xaxis={"title": "Minutes from now", "range": [min(-56, min(x_past) - 1), 31],
               "zeroline": False, "tickfont": {"color": "#52677D"}},
        yaxis={"title": y_label, "rangemode": "tozero", "gridcolor": "#EBEEF1",
               "tickfont": {"color": "#52677D"}},
        legend={"orientation": "h", "y": 1.13, "x": 0},
    )
    st.plotly_chart(figure, width="stretch", key="qm_forecast_chart")
    context = "simulated live history" if use_live_queue else "scenario-adjusted context"
    st.caption(f"SYNTHETIC {context} · deterministic queue simulation, not a trained-model prediction.")


def _counter_cards(state, view):
    options = [item for item in view["options"] if item["active_counters"] <= 6]
    max_wait = max(item["predicted_wait_minutes"] for item in options) or 1
    columns = st.columns(len(options))
    recommended = view["recommendation"]["recommended_counters"]
    for column, option in zip(columns, options):
        count = option["active_counters"]
        risk = risk_for(option["future_queue"], option["predicted_wait_minutes"], view["config"])
        role = "current" if count == state["active_counters"] else "recommended" if count == recommended else ""
        flag = "CURRENT" if count == state["active_counters"] else "★ RECOMMENDED" if count == recommended else "CAPACITY OPTION"
        width = max(4, round(option["predicted_wait_minutes"] / max_wait * 100))
        with column:
            st.markdown(
                f'<div class="qm-whatif {role}"><div class="qm-label">{flag}</div>'
                f'<div class="big">{count} <span class="qm-muted">counters</span></div>'
                f'<div><b>{option["predicted_wait_minutes"]:.1f} min</b> · '
                f'{option["future_queue"]:.0f} people</div>'
                f'<span class="qm-badge {risk}">{risk.upper()}</span>'
                f'<div class="qm-mini-bar"><i style="width:{width}%"></i></div>'
                f'<div class="qm-muted">{count} staffing slot{"s" if count != 1 else ""} · +20 min</div></div>',
                unsafe_allow_html=True,
            )
            clicked = st.button(
                "Apply" if count != state["active_counters"] else "Selected",
                key=f"qm_option_{count}", width="stretch",
                disabled=count == state["active_counters"],
                on_click=_queue_counter_option, args=(count,),
            )
            if clicked:
                st.rerun()
    previous_count = st.session_state.get("qm_comparison_from")
    if previous_count is not None and previous_count != state["active_counters"]:
        before_count, after_count = previous_count, state["active_counters"]
        heading = "PREVIOUS → SELECTED / SIMULATED +20 MIN"
        before, after = evaluate_counter_counts(
            state["current_queue"], view["arrival_rate"],
            view["config"]["service_time"], [before_count, after_count], 20,
        )
        outcomes = {item["active_counters"]: item for item in (before, after)}
        before, after = outcomes[before_count], outcomes[after_count]
    else:
        before_count, after_count = state["active_counters"], recommended
        heading = "CURRENT → RECOMMENDED / SIMULATED +20 MIN"
        before = view["recommendation"]["current_outcome"]
        after = view["recommendation"]["recommended_outcome"]
    change = (
        (after["predicted_wait_minutes"] - before["predicted_wait_minutes"])
        / before["predicted_wait_minutes"] * 100
        if before["predicted_wait_minutes"] else 0
    )
    if change < 0:
        impact = f"Wait decreases by {-change:.0f}%"
    elif change > 0:
        impact = f"Wait increases by {change:.0f}%"
    else:
        impact = "Wait is unchanged"
    st.markdown(
        f'<div class="qm-panel mint"><div class="qm-label">{heading}</div>'
        f'<b>{before_count} counters</b> · {before["future_queue"]:.0f} people · '
        f'{before["predicted_wait_minutes"]:.1f} min wait &nbsp; → &nbsp; '
        f'<b>{after_count} counters</b> · {after["future_queue"]:.0f} people · '
        f'{after["predicted_wait_minutes"]:.1f} min wait'
        f'<div class="qm-muted">{impact}. '
        'These are deterministic simulation outcomes, not measured results.</div></div>',
        unsafe_allow_html=True,
    )


def _queue_counter_option(count):
    # Fragment buttons need the outer slider to be reconstructed on an app
    # rerun; otherwise its stale widget value can undo the chosen capacity.
    st.session_state.qm_pending_counters = count


def _location_strip(locations, selected):
    by_id = {location["id"]: location for location in locations}
    columns = st.columns(len(SCENARIO_OPTIONS))
    for column, (scenario, _) in zip(columns, SCENARIO_OPTIONS):
        scenario_state = st.session_state.qm_live if scenario == selected else initial_state(scenario)
        view = snapshot(scenario_state)
        location = by_id.get(view["config"]["location_id"])
        label = location["display_name"] if location else scenario
        with column:
            st.button(
                f'{label}  ·  {view["queue"]:.0f} people  ·  {view["wait"]:.0f}m',
                key=f"qm_location_{scenario}", width="stretch",
                type="primary" if scenario == selected else "secondary",
                on_click=_select_scenario, args=(scenario,),
            )
    st.caption("All facility snapshots are SYNTHETIC scenario states, not live campus measurements.")


def _heatmap(location, state):
    metric = st.radio(
        "Heatmap metric", ("Queue", "Wait"), horizontal=True,
        key="qm_heatmap_metric",
    )
    history = _history(location, 14, 11)
    matrix = heatmap_data(
        history, metric, SCENARIOS[state["selected_scenario"]]["service_time"],
        state["active_counters"],
    )
    figure = go.Figure(go.Heatmap(
        z=matrix.values, x=list(WEEKDAYS), y=list(TIME_SLOTS),
        colorscale=[[0, "#EDF3F8"], [.45, "#F6C9A5"], [1, "#F36F5D"]],
        colorbar={"title": "People" if metric == "Queue" else "Minutes"},
        hovertemplate="%{x} · %{y}<br>Average %{z:.1f}<extra>SYNTHETIC</extra>",
    ))
    figure.update_layout(
        height=280, margin={"l": 5, "r": 5, "t": 8, "b": 5},
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font={"family": "Manrope, sans-serif", "color": "#52677D", "size": 11},
        xaxis={"tickfont": {"color": "#52677D", "size": 11}},
        yaxis={"autorange": "reversed", "tickfont": {"color": "#52677D", "size": 11}},
    )
    st.plotly_chart(figure, width="stretch", key="qm_heatmap_chart")
    st.caption("SYNTHETIC 14-day weekday averages from the configured location profile. No campus observations.")


def _vision_card():
    local_ready = all(importlib.util.find_spec(name) for name in ("cv2", "ultralytics"))
    status = "Local dependencies available" if local_ready else "Install local vision dependencies to run"
    st.markdown(
        '<div class="qm-panel blue"><div class="qm-label">VISION PIPELINE / OPTIONAL LOCAL MODULE</div>'
        f'<h3>{status}</h3><div class="qm-flow">'
        '<span>VIDEO</span><span>PERSON DETECTION</span><span>ANONYMOUS TRACKING</span>'
        '<span>QUEUE ROI</span><span>QUEUE STATS</span></div>'
        '<p class="qm-muted">Cloud live camera: LOCAL ONLY. No face recognition, identity tracking, '
        'or raw-video retention by default. Quantitative queue-count accuracy: not evaluated yet.</p></div>',
        unsafe_allow_html=True,
    )


def _model_card(locations):
    try:
        metrics = _evaluation(tuple(locations))
    except (ValueError, RuntimeError, ImportError) as exc:
        st.info(f"Forecast evaluation not available: {exc}")
        return
    improvements = ml_improvement(metrics)
    columns = st.columns(4)
    for column, horizon in zip(columns, HORIZONS):
        values = metrics[horizon]
        with column:
            st.markdown(
                '<div class="qm-panel lilac">'
                f'<div class="qm-label">+{horizon} MIN · HELD-OUT SYNTHETIC</div>'
                f'<div class="qm-number">{values["ml"]["mae"]:.2f} <small>MAE</small></div>'
                f'<div class="qm-muted">Baseline MAE {values["baseline"]["mae"]:.2f}'
                f' · {values["test_samples"]} samples</div></div>',
                unsafe_allow_html=True,
            )
    figure = go.Figure()
    for key, label, color in (
        ("baseline", "Rolling baseline", "#173B63"),
        ("ml", "Gradient boosting", "#F36F5D"),
    ):
        figure.add_trace(go.Bar(
            name=label, x=[f"+{h} min" for h in HORIZONS],
            y=[metrics[h][key]["rmse"] for h in HORIZONS],
            marker_color=color,
        ))
    figure.update_layout(
        barmode="group", height=240, margin={"l": 10, "r": 5, "t": 5, "b": 10},
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        yaxis_title="RMSE · arrivals / interval",
        font={"family": "Manrope, sans-serif", "color": "#52677D", "size": 11},
        legend={"orientation": "h", "y": 1.15},
    )
    st.plotly_chart(figure, width="stretch", key="qm_model_chart")
    delta = improvements.get(30)
    comparison = (
        f"+30 min ML RMSE change vs baseline: {delta:+.1f}% improvement"
        if delta is not None else "Baseline RMSE is zero; percentage comparison unavailable."
    )
    st.caption(
        "SYNTHETIC HELD-OUT EVALUATION · " + comparison
        + " · Results are not real-campus accuracy. Forecast chart above uses deterministic scenario projection."
    )


def render_command_center(locations):
    """Compose the page; only lightweight live content reruns each second."""
    if "qm_live" not in st.session_state:
        st.session_state.qm_live = initial_state("Canteen Lunch Rush")
        _sync_controls()
    elif "qm_counter_version" not in st.session_state:
        _sync_controls()
    state = st.session_state.qm_live
    pending = st.session_state.pop("qm_pending_counters", None)
    if pending is not None:
        st.session_state.qm_comparison_from = state["active_counters"]
        set_counters(state, pending)
        _rotate_counter_widget()
    top_nav(state)
    hero(state["selected_scenario"], state)
    st.markdown('<div class="qm-label">SCENARIO COMMAND BAR · SYNTHETIC SIMULATION</div>', unsafe_allow_html=True)
    scenario_columns = st.columns(4)
    for column, (name, label) in zip(scenario_columns, SCENARIO_OPTIONS):
        with column:
            st.button(
                label, key=f"qm_scenario_{name}", width="stretch",
                type="primary" if state["selected_scenario"] == name else "secondary",
                on_click=_select_scenario, args=(name,),
            )
    controls = st.columns([1.25, 1, 1, .7, .7, .7, 1])
    controls[0].slider("Demand intensity", .5, 2.5, step=.1, key="qm_demand", on_change=_change_demand)
    controls[1].slider(
        "Active counters", 1, 8,
        key=f'qm_counters_{st.session_state.qm_counter_version}',
        on_change=_change_counters,
    )
    controls[2].selectbox("Simulation speed", list(SPEEDS), key="qm_speed", on_change=_change_speed)
    controls[3].button("▶ PLAY", width="stretch", on_click=play, args=(state,))
    controls[4].button("Ⅱ PAUSE", width="stretch", on_click=pause, args=(state,))
    controls[5].button("↺ RESET", width="stretch", on_click=_reset)
    controls[6].button("⚡ TRIGGER SURGE", width="stretch", on_click=_surge, type="primary")

    location_by_id = {location["id"]: location for location in locations}
    location = location_by_id[state["selected_location"]]
    history = _history(location, 1, 42)

    @st.fragment(run_every="1s")
    def live_sections():
        current = st.session_state.qm_live
        advance(current)
        view = snapshot(current)
        st.caption(
            ("● LIVE SIMULATION" if current["simulation_running"] else "○ PAUSED / READY")
            + f' · +{current["simulation_elapsed_minutes"]} simulated minutes · '
            "SYNTHETIC, not live campus data"
        )
        kpi_strip(view, current)
        section("live", "Live queue & recommendation", "Current synthetic state → operational action")
        left, right = st.columns([1.7, 1.15])
        with left:
            live_queue(view, current)
        with right:
            action_center(view, current)
            recommended = view["recommendation"]["recommended_counters"]
            if recommended != current["active_counters"]:
                clicked = st.button(
                    f"SIMULATE {recommended} COUNTERS", type="primary",
                    key="qm_apply_recommendation", width="stretch",
                    on_click=_queue_counter_option, args=(recommended,),
                )
                if clicked:
                    st.rerun()
        section("forecast", "Now → future", "Deterministic projection · 5 to 30 minutes")
        timeline(view)
        chart_col, feed_col = st.columns([1.8, 1])
        with chart_col:
            _forecast_chart(current, view, history)
        with feed_col:
            st.markdown('<div class="qm-label">PREDICTIVE ALERTS</div>', unsafe_allow_html=True)
            filter_name = st.radio(
                "Alert filter", ("ALL", "CRITICAL", "WARNING"),
                horizontal=True, label_visibility="collapsed", key="qm_alert_filter",
            )
            items = alert_items(current, view)
            if filter_name != "ALL":
                items = [item for item in items if item["level"] == filter_name.lower()]
            alert_feed(items)
            st.markdown('<div class="qm-label" style="margin-top:8px">LIVE EVENT STREAM</div>', unsafe_allow_html=True)
            event_stream(current["simulation_events"])
        section("what-if", "Counter what-if simulator", "Choose capacity · compare future outcomes")
        _counter_cards(current, view)

    live_sections()
    section("campus", "Campus location strip", "Click a facility to change the active simulation")
    _location_strip(locations, st.session_state.qm_live["selected_scenario"])
    section("heatmap", "Congestion heatmap", "Weekday synthetic queue patterns")
    active = st.session_state.qm_live
    _heatmap(location_by_id[active["selected_location"]], active)
    section("vision", "Vision system", "Optional local-only analytics")
    _vision_card()
    section("model", "ML model evaluation", "Chronological synthetic held-out test")
    st.caption("No model is trained during live ticks.")
    _model_card(tuple(locations))
    st.caption("QueueMind · Academic prototype · All displayed queue states are SYNTHETIC SIMULATION.")
