"""Reusable presentation fragments for the synthetic Command Center."""

from html import escape

import streamlit as st


def _show(markup):
    st.markdown(markup, unsafe_allow_html=True)


def top_nav(state):
    sections = (
        ("overview", "OVERVIEW"),
        ("live", "LIVE QUEUE"),
        ("forecast", "FORECAST"),
        ("what-if", "WHAT-IF"),
        ("heatmap", "HEATMAP"),
        ("vision", "VISION"),
        ("model", "MODEL"),
    )
    links = "".join(
        f'<a href="#{anchor}">{label}</a>' for anchor, label in sections
    )
    active = state["simulation_running"]
    status = "LIVE SIMULATION · SYNTHETIC" if active else "SYNTHETIC DEMO"
    dot_class = "qm-live-dot running" if active else "qm-live-dot"
    _show(
        '<nav class="qm-nav" aria-label="Command Center sections">'
        '<div class="qm-brand"><span class="qm-clock">◷</span>QueueMind</div>'
        f'<div class="qm-links">{links}</div>'
        f'<div class="qm-nav-status"><span class="{dot_class}"></span>'
        f'{status}</div></nav>'
    )


def section(anchor, title, detail=""):
    _show(
        f'<div id="{anchor}" class="qm-section-anchor qm-section-head">'
        f'<h2>{escape(title)}</h2><span>{escape(detail)}</span></div>'
    )


def hero(scenario, state):
    status = (
        "● LIVE SIMULATION" if state["simulation_running"]
        else "● SYNTHETIC DEMO"
    )
    _show(
        '<div id="overview" class="qm-section-anchor qm-hero">'
        '<div><div class="qm-eyebrow">QUEUE INTELLIGENCE / COMMAND CENTER</div>'
        '<div class="qm-title">See now. Predict next. Act early.</div>'
        f'<div class="qm-subtitle">{escape(scenario)} · '
        'All queue values are synthetic simulation data.</div></div>'
        f'<span class="qm-tag">{status}</span></div>'
    )


def kpi_strip(view, state):
    config = view["config"]
    items = (
        ("CURRENT QUEUE", f'{view["queue"]:.0f} <small>people</small>',
         f'{view["queue20"] - view["queue"]:+.0f} expected in 20 min'),
        ("ESTIMATED WAIT", f'{view["wait"]:.1f} <small>min</small>',
         f'Target {config["target_wait"]} min'),
        ("+20 MIN QUEUE", f'{view["queue20"]:.0f} <small>people</small>',
         "Forecast / simulated"),
        ("ACTIVE COUNTERS", f'{state["active_counters"]}',
         f'Recommended {view["recommendation"]["recommended_counters"]}'),
        ("QUEUE PRESSURE", f'{view["pressure"]:+.2f} <small>/min</small>',
         "Arrivals − service capacity"),
    )
    cards = "".join(
        '<div class="qm-kpi">'
        f'<div class="qm-label">{label}</div><div class="qm-number">{value}</div>'
        f'<div class="qm-help">{detail}</div></div>'
        for label, value, detail in items
    )
    _show(f'<div class="qm-kpis">{cards}</div>')


def live_queue(view, state):
    count = max(0, round(view["queue"]))
    visible = min(count, 40)
    dots = "".join(
        '<span class="qm-person" aria-hidden="true"></span>'
        for _ in range(visible)
    )
    extra = f'<span class="qm-muted">+{count - 40} more</span>' if count > 40 else ""
    counters = "".join(
        f'<span class="qm-counter">● &nbsp; COUNTER {n}</span>'
        for n in range(1, state["active_counters"] + 1)
    )
    serving = "".join(
        '<i aria-hidden="true"></i>'
        for _ in range(state["active_counters"])
    )
    _show(
        '<div class="qm-panel blue">'
        '<div class="qm-label">LIVE QUEUE / ANONYMOUS PEOPLE</div>'
        f'<div style="display:flex;align-items:center;justify-content:space-between">'
        f'<b style="font-size:18px">{count} waiting</b>'
        f'<span class="qm-muted">{state["active_counters"]} serving counters</span></div>'
        '<div class="qm-queue-flow">'
        f'<div class="qm-dots">{dots or "Queue is clear."}{extra}</div>'
        '<div class="qm-service-stage">→ <div><small>BEING SERVED</small>'
        f'<div class="qm-service-dots">{serving}</div></div></div></div>'
        f'<div class="qm-counter-list">{counters}</div>'
        '<div class="qm-visual-key"><span>● Waiting</span>'
        '<span style="color:#F36F5D">● Being served</span>'
        '<span style="color:#2DA36B">● Open counter</span></div></div>'
    )


def action_center(view, state):
    current = view["recommendation"]["current_outcome"]
    recommended = view["recommendation"]["recommended_outcome"]
    target = view["config"]["target_wait"]
    current_counters = state["active_counters"]
    new_counters = view["recommendation"]["recommended_counters"]
    if new_counters > current_counters:
        headline = f"Open {new_counters - current_counters} additional counter"
        if new_counters - current_counters != 1:
            headline += "s"
    elif new_counters < current_counters:
        headline = f"{new_counters} counters can meet the target"
    else:
        headline = f"Keep {current_counters} counters open"
    threshold = view["first_target_exceeded"]
    if threshold == 0:
        explanation = f"Current wait already exceeds the {target}-minute target."
    elif threshold is not None:
        explanation = f"Wait crosses the {target}-minute target by +{threshold} min."
    else:
        explanation = f"Simulated wait remains within the {target}-minute target."
    risk = view["risk20"]
    _show(
        '<div class="qm-panel coral">'
        f'<div class="qm-label">QUEUE MIND RECOMMENDATION &nbsp; '
        f'<span class="qm-badge {risk}">{risk.upper()} RISK</span></div>'
        f'<div class="qm-action-title">{headline}</div>'
        f'<div class="qm-action-meta">{explanation}</div>'
        '<div class="qm-impact">'
        f'<div><span class="qm-label">CURRENT · +20M</span>'
        f'<b>{current["predicted_wait_minutes"]:.1f} min</b>'
        f'<span class="qm-muted">{current["future_queue"]:.0f} people</span></div>'
        f'<div><span class="qm-label">RECOMMENDED · +20M</span>'
        f'<b>{recommended["predicted_wait_minutes"]:.1f} min</b>'
        f'<span class="qm-muted">{recommended["future_queue"]:.0f} people</span></div>'
        '</div></div>'
    )


def timeline(view):
    threshold = view["first_target_exceeded"]
    nodes = []
    for item in view["timeline"]:
        minute = item["minute"]
        label = "NOW" if minute == 0 else f"+{minute} MIN"
        special = " exceeded" if minute == threshold else ""
        flag = '<small>TARGET EXCEEDED</small>' if minute == threshold else ""
        nodes.append(
            f'<div class="qm-time-node{special}"><div class="qm-label">{label}</div>'
            f'<strong>{item["queue"]:.0f}</strong> '
            f'<span class="qm-muted">people</span>'
            f'<small>{item["wait"]:.1f} min wait</small>'
            f'<span class="qm-badge {item["risk"]}">{item["risk"].upper()}</span>'
            f'{flag}</div>'
        )
    _show(f'<div class="qm-timeline">{"".join(nodes)}</div>')


def alert_feed(items):
    if not items:
        _show('<div class="qm-panel mint">No predictive alerts at current capacity.</div>')
        return
    rows = "".join(
        '<div class="qm-alert">'
        f'<span class="qm-badge {item["level"]}">{item["level"].upper()}</span>'
        f'<b>{escape(item["title"])}</b>'
        f'<p>{escape(item["detail"])}</p></div>'
        for item in items
    )
    _show(f'<div class="qm-panel amber">{rows}</div>')


def event_stream(events):
    if not events:
        _show('<div class="qm-panel">Press Play or Trigger Surge to begin the event stream.</div>')
        return
    rows = "".join(
        f'<div class="qm-event"><time>+{item["minute"]}m</time>'
        f'<span>{escape(item["text"])}</span></div>'
        for item in reversed(events[-6:])
    )
    _show(f'<div class="qm-panel">{rows}</div>')
