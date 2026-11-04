"""Overview, live, forecast, optimization, vision, and evidence surfaces."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.demo.live import SPEEDS, advance, pause, play
from src.demo.scenarios import SCENARIOS
from src.queueing.simulator import calculate_service_rate, recommend_counter_count
from src.ui.command_center import (
    SCENARIO_OPTIONS, _change_counters, _change_demand, _change_speed,
    _counter_cards, _forecast_chart, _heatmap, _history, _queue_counter_option,
    _reset, _select_scenario, _surge, _vision_card,
)
from src.ui.command_model import HORIZONS, alert_items, risk_for, snapshot
from src.ui.command_views import (
    action_center, alert_feed, event_stream, kpi_strip, live_queue, timeline,
)


def _controls(state):
    st.caption("SYNTHETIC SCENARIO CONTROLS")
    choices = st.columns(4)
    for column, (name, label) in zip(choices, SCENARIO_OPTIONS):
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
    controls[6].button("⚡ SURGE", width="stretch", on_click=_surge, type="primary")


def _recommendation_button(state, view):
    count = view["recommendation"]["recommended_counters"]
    if count != state["active_counters"]:
        clicked = st.button(
            f"SIMULATE {count} COUNTERS", key="qm_apply_recommendation",
            type="primary", width="stretch", on_click=_queue_counter_option,
            args=(count,),
        )
        if clicked:
            st.rerun()


def _location(state, locations):
    return next(item for item in locations if item["id"] == state["selected_location"])


def render_overview_tab(locations):
    if st.session_state.get("qm_state_source") == "VISION":
        from src.ui.vision_lab import render_shared_vision_state
        if render_shared_vision_state(locations, "Overview"):
            return
    state = st.session_state.qm_live
    _controls(state)

    @st.fragment(run_every="1s")
    def overview_live():
        current = st.session_state.qm_live
        advance(current)
        view = snapshot(current)
        st.caption("SYNTHETIC DEMO · " + ("LIVE" if current["simulation_running"] else "READY / PAUSED"))
        kpi_strip(view, current)
        left, right = st.columns([1.7, 1.15])
        with left:
            live_queue(view, current)
        with right:
            action_center(view, current)
            _recommendation_button(current, view)
        st.markdown("#### Now → future · simulated forecast")
        timeline(view)
        alerts = alert_items(current, view)[:3]
        st.markdown("#### Predictive alerts")
        alert_feed(alerts)
        st.markdown("#### Counter what-if · +20 min")
        _counter_cards(current, view)

    overview_live()
    st.markdown("#### Synthetic congestion preview")
    _heatmap(_location(st.session_state.qm_live, locations), st.session_state.qm_live)
    ml_col, vision_col = st.columns(2)
    with ml_col:
        st.markdown("#### ML snapshot")
        st.markdown(
            '<div class="qm-panel lilac"><b>HistGradientBoostingRegressor</b>'
            '<p>Separate +5 / +10 / +20 / +30 minute arrival models. '
            'Rolling-arrival baseline. Chronological synthetic evaluation.</p></div>',
            unsafe_allow_html=True,
        )
        if st.session_state.get("qm_ml_ready"):
            from src.ui.ml_service import evaluation_bundle

            metrics = evaluation_bundle(tuple(locations))["metrics"]
            st.caption(
                "SYNTHETIC held-out +20m MAE · ML "
                f'{metrics[20]["ml"]["mae"]:.2f} vs baseline {metrics[20]["baseline"]["mae"]:.2f}'
            )
        else:
            st.caption("Metrics pending · open ML Lab to run evaluation explicitly.")
        if st.button("OPEN ML LAB", key="qm_open_ml_lab"):
            st.session_state.main_tabs = "ML Lab"
            st.rerun()
    with vision_col:
        st.markdown("#### Vision snapshot")
        _vision_card()
    st.caption("All queue values above are SYNTHETIC SIMULATION, not measured campus operations.")


def render_live_tab(locations):
    if st.session_state.get("qm_state_source") == "VISION":
        st.info("This Live Simulation tab remains SYNTHETIC and is independent of the selected local-video state. Overview, Forecast, and Optimizer use the selected video state.")
    state = st.session_state.qm_live
    _controls(state)

    @st.fragment(run_every="1s")
    def live_detail():
        current = st.session_state.qm_live
        advance(current)
        view = snapshot(current)
        st.caption(f'SYNTHETIC LIVE SIMULATION · +{current["simulation_elapsed_minutes"]} simulated minutes')
        kpi_strip(view, current)
        left, right = st.columns([1.7, 1.15])
        with left:
            live_queue(view, current)
        with right:
            action_center(view, current)
            _recommendation_button(current, view)
        chart_col, feed_col = st.columns([1.7, 1])
        with chart_col:
            _live_charts(current, view)
        with feed_col:
            st.markdown("#### Event stream")
            event_stream(current["simulation_events"])
            st.markdown("#### Predictive alerts")
            alert_feed(alert_items(current, view)[:5])

    live_detail()


def _live_charts(state, view):
    points = state.get("queue_history", [])
    if len(points) < 2:
        st.info("Press Play to populate simulated queue and wait history.")
        return
    minutes = [item["minute"] for item in points]
    queues = [item["queue"] for item in points]
    waits = [queue / view["service_rate"] for queue in queues]
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=minutes, y=queues, name="Queue", line_color="#173B63"))
    chart.add_trace(go.Scatter(x=minutes, y=waits, name="Wait · min", line_color="#F36F5D"))
    chart.update_layout(height=250, margin=dict(l=10, r=10, t=15, b=10), xaxis_title="Simulated minute")
    st.plotly_chart(chart, width="stretch", key="qm_live_history")
    rate_chart = go.Figure()
    rate_chart.add_trace(go.Bar(x=["Arrival rate", "Service rate"],
                                y=[view["arrival_rate"], view["service_rate"]],
                                marker_color=["#F36F5D", "#2DA36B"]))
    rate_chart.update_layout(height=180, margin=dict(l=10, r=10, t=5, b=10),
                             yaxis_title="People / min")
    st.plotly_chart(rate_chart, width="stretch", key="qm_live_rates")
    st.caption("SYNTHETIC rate snapshot · historical wait uses current capacity for visual comparison.")


def render_forecast_tab(locations):
    if st.session_state.get("qm_state_source") == "VISION":
        from src.ui.vision_lab import render_shared_vision_state
        if render_shared_vision_state(locations, "Forecast"):
            return
    state = st.session_state.qm_live
    view = snapshot(state)
    st.markdown("### Forecast · deterministic scenario projection")
    st.caption("SYNTHETIC scenario demand assumption. This chart is not an ML-model prediction.")
    timeline(view)
    history = _history(_location(state, locations), 1, 42)
    _forecast_chart(state, view, history)
    rows = []
    for point in view["timeline"][1:]:
        horizon = point["minute"]
        recommendation = recommend_counter_count(
            state["current_queue"], view["arrival_rate"],
            view["config"]["service_time"], state["active_counters"],
            view["config"]["target_wait"], horizon, list(range(1, 9)),
        )
        rows.append({
            "Horizon": f"+{horizon} min", "Arrival rate / min": view["arrival_rate"],
            "Service rate / min": view["service_rate"],
            "Predicted queue": point["queue"], "Predicted wait · min": point["wait"],
            "Risk": point["risk"].upper(),
            "Counters required": recommendation["recommended_counters"],
        })
    table = pd.DataFrame(rows)
    st.markdown("#### Horizon detail")
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("DOWNLOAD SYNTHETIC FORECAST CSV", table.to_csv(index=False),
                       "queuemind_synthetic_forecast.csv", "text/csv")
    st.markdown("#### Why is the queue changing?")
    pressure = view["pressure"]
    st.write(
        f'Arrival rate **{view["arrival_rate"]:.2f} people/min** − service capacity '
        f'**{view["service_rate"]:.2f} people/min** = **{pressure:+.2f} people/min**.'
    )
    st.info("Arrivals exceed current service capacity." if pressure > 0
            else "Current service capacity meets the scenario demand assumption.")
    st.caption("This explains queue mathematics, not ML feature importance or causality.")


def render_optimizer_tab(locations):
    if st.session_state.get("qm_state_source") == "VISION":
        from src.ui.vision_lab import render_shared_vision_state
        if render_shared_vision_state(locations, "Optimizer"):
            return
    state = st.session_state.qm_live
    view = snapshot(state)
    st.markdown("### Counter optimizer · simulated decision support")
    st.caption("SYNTHETIC +20-minute outcomes · click Apply to update the shared simulation state.")
    _counter_cards(state, view, full=True)
    current = view["recommendation"]["current_outcome"]
    proposed = view["recommendation"]["recommended_outcome"]
    current_rate = calculate_service_rate(state["active_counters"], view["config"]["service_time"])
    proposed_rate = calculate_service_rate(view["recommendation"]["recommended_counters"], view["config"]["service_time"])
    left, right = st.columns(2)
    with left:
        st.markdown("#### Current configuration")
        st.metric("Counters", state["active_counters"])
        st.metric("+20 queue", f'{current["future_queue"]:.0f}')
        st.metric("+20 wait", f'{current["predicted_wait_minutes"]:.1f} min')
        st.metric("Risk", risk_for(current["future_queue"], current["predicted_wait_minutes"], view["config"]).upper())
        st.caption(f"Service capacity {current_rate:.2f} people/min")
    with right:
        st.markdown("#### Recommended configuration")
        st.metric("Counters", view["recommendation"]["recommended_counters"])
        st.metric("+20 queue", f'{proposed["future_queue"]:.0f}')
        st.metric("+20 wait", f'{proposed["predicted_wait_minutes"]:.1f} min')
        st.metric("Risk", risk_for(proposed["future_queue"], proposed["predicted_wait_minutes"], view["config"]).upper())
        st.caption(f"Service capacity {proposed_rate:.2f} people/min")
    st.markdown("#### Why this number of counters?")
    st.write(view["recommendation"]["reason"])
    if current["future_queue"]:
        st.metric("Simulated queue reduction", f'{(current["future_queue"] - proposed["future_queue"]) / current["future_queue"]:.1%}')
    if current["predicted_wait_minutes"]:
        st.metric("Simulated wait reduction", f'{(current["predicted_wait_minutes"] - proposed["predicted_wait_minutes"]) / current["predicted_wait_minutes"]:.1%}')
    st.caption(
        f'Target wait {view["config"]["target_wait"]} min · '
        f'forecast demand {view["arrival_rate"]:.2f} people/min · '
        "the optimizer chooses the minimum tested counter count meeting the target. "
        "Additional counters may reduce wait further but use more staffing slots."
    )


def render_vision_tab():
    from src.ui.vision_lab import render_vision_lab
    render_vision_lab()


def render_evaluation_tab(locations):
    st.markdown("### Evaluation · quantitative evidence")
    st.markdown("#### Forecasting · held-out SYNTHETIC observations")
    if st.session_state.get("qm_ml_ready"):
        from src.ui.ml_lab import metric_table
        from src.ui.ml_service import evaluation_bundle

        bundle = evaluation_bundle(tuple(locations))
        st.dataframe(metric_table(bundle["metrics"]), hide_index=True, width="stretch")
        horizon = st.select_slider("Actual vs predicted horizon", options=[5, 10, 20, 30], value=20)
        frame = bundle["forecaster"].test_predictions[horizon].tail(240)
        figure = go.Figure()
        for column, label, color in (("actual", "Actual", "#173B63"), ("predicted", "ML", "#6B4EAA"), ("baseline", "Baseline", "#F36F5D")):
            figure.add_trace(go.Scatter(x=frame.timestamp, y=frame[column], name=label, line_color=color))
        figure.update_layout(height=300, title="Latest 240 held-out synthetic test rows · all facilities")
        st.plotly_chart(figure, width="stretch")
    else:
        st.info("Forecast evaluation not run in this session. Open ML Lab → Overview and run it explicitly.")
    st.markdown("#### Vision · manually labelled frame counts")
    metrics = st.session_state.get("qm_vision_evaluation")
    if metrics:
        st.dataframe(pd.DataFrame([metrics]), hide_index=True, width="stretch")
        st.caption("Calculated from user-supplied labels. Real-video provenance is not verified by the app.")
    else:
        st.info("Real-video quantitative evaluation pending. Upload labels in Vision Lab.")
    st.markdown("#### System · SYNTHETIC counter what-if")
    state = st.session_state.qm_live
    view = snapshot(state)
    current, proposed = view["recommendation"]["current_outcome"], view["recommendation"]["recommended_outcome"]
    comparison = pd.DataFrame([
        {"Configuration": "Current", **current},
        {"Configuration": "Recommended", **proposed},
    ])
    st.dataframe(comparison, hide_index=True, width="stretch")
    queue_reduction = current["future_queue"] - proposed["future_queue"]
    wait_reduction = current["predicted_wait_minutes"] - proposed["predicted_wait_minutes"]
    st.caption(f"Simulated +20-minute reduction: queue {queue_reduction:+.1f} people; wait {wait_reduction:+.1f} minutes. Negative values mean the proposed state is worse. These are deterministic what-if outcomes, not measured intervention effects.")
