"""Scoped visual system for the single-page QueueMind Command Center."""

import streamlit as st


def apply_command_theme():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
:root {
  --qm-bg:#F7F4EF; --qm-white:#FFFFFF; --qm-navy:#173B63; --qm-ink:#102033;
  --qm-coral:#F36F5D; --qm-blue:#3478F6; --qm-teal:#159A82; --qm-green:#2DA36B;
  --qm-amber:#E5A128; --qm-red:#D84E4E; --qm-muted:#718096; --qm-border:#DFE5EA;
}
html { scroll-behavior:smooth; }
html, body, [data-testid="stAppViewContainer"] {
  background:var(--qm-bg); color:var(--qm-ink);
  font-family:Manrope,Inter,"Segoe UI",sans-serif;
}
[data-testid="stSidebar"], [data-testid="stHeader"], footer {display:none;}
.block-container {max-width:1550px;padding:0.65rem 1.7rem 2rem;}
[data-testid="stVerticalBlock"] {gap:0.48rem;}
[data-testid="stElementContainer"]:has(.qm-nav) {position:sticky;top:0;z-index:1000;}
h2, h3, p {color:var(--qm-ink);}
h3 {font-size:1.15rem!important;margin:0.25rem 0!important;}
button {transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease!important;}
button:hover {transform:translateY(-1px);box-shadow:0 5px 16px rgba(23,59,99,.09);}
button[kind="primary"] {background:var(--qm-coral)!important;border-color:var(--qm-coral)!important;color:white!important;}
button[kind="secondary"] {background:white!important;border-color:var(--qm-border)!important;color:var(--qm-navy)!important;}
div[data-baseweb="select"]>div {background:white!important;border-color:var(--qm-border)!important;}
.qm-nav {position:sticky;top:0;z-index:1000;display:flex;align-items:center;gap:18px;
  min-height:55px;padding:8px 14px;margin:-.65rem -.2rem .5rem;
  border:1px solid var(--qm-border);border-radius:0 0 15px 15px;
  background:rgba(255,255,255,.97);box-shadow:0 5px 22px rgba(23,59,99,.06);}
.qm-brand {font-size:17px;font-weight:800;color:var(--qm-navy);white-space:nowrap;}
.qm-clock {display:inline-flex;align-items:center;justify-content:center;width:27px;height:27px;
  border:2px solid var(--qm-navy);border-radius:50%;color:var(--qm-coral);margin-right:7px;}
.qm-links {display:flex;gap:3px;flex:1;overflow:auto;scrollbar-width:none;}
.qm-links a {padding:6px 9px;border-radius:8px;color:#52677D;text-decoration:none!important;
  white-space:nowrap;font-size:11px;font-weight:800;letter-spacing:.035em;}
.qm-links a:hover,.qm-links a:focus {background:#FFF0EB;color:var(--qm-coral);}
.qm-nav-status {font-size:10px;color:var(--qm-navy);font-weight:800;white-space:nowrap;}
.qm-live-dot {display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--qm-green);margin-right:5px;}
.qm-live-dot.running {animation:qm-pulse 2s ease-in-out infinite;}
@keyframes qm-pulse {50% {opacity:.45;}}
.qm-section-anchor {scroll-margin-top:70px;}
.qm-hero {display:flex;align-items:flex-end;justify-content:space-between;gap:14px;margin:2px 0 8px;}
.qm-eyebrow {font-size:10px;color:var(--qm-coral);font-weight:800;letter-spacing:.13em;}
.qm-title {font-size:25px;font-weight:800;line-height:1.14;letter-spacing:-.045em;margin:2px 0;color:var(--qm-ink);}
.qm-subtitle {font-size:12px;color:var(--qm-muted);}
.qm-tag {background:#FFF0EB;color:#B6493A;border-radius:99px;padding:7px 10px;font-size:10px;font-weight:800;white-space:nowrap;}
.qm-section-head {display:flex;align-items:end;justify-content:space-between;margin:14px 0 7px;}
.qm-section-head h2 {font-size:18px!important;letter-spacing:-.025em;margin:0!important;}
.qm-section-head span {font-size:11px;color:var(--qm-muted);}
.qm-panel {background:white;border:1px solid var(--qm-border);border-radius:15px;
  box-shadow:0 6px 20px rgba(23,59,99,.045);padding:13px 15px;}
.qm-panel.blue {background:#EDF3F8;}
.qm-panel.coral {background:#FFF0EB;}
.qm-panel.mint {background:#EAF7F0;}
.qm-panel.amber {background:#FFF6E5;}
.qm-panel.lilac {background:#F4F0FF;}
.qm-kpis {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px;margin:4px 0 6px;}
.qm-kpi {border:1px solid var(--qm-border);border-radius:13px;padding:12px 14px;min-width:0;background:#fff;}
.qm-kpi:nth-child(1){background:#EDF3F8}.qm-kpi:nth-child(2){background:#FFF0EB}
.qm-kpi:nth-child(3){background:#F4F0FF}.qm-kpi:nth-child(4){background:#EAF7F0}
.qm-kpi:nth-child(5){background:#FFF6E5}
.qm-label {font-size:10px;font-weight:800;letter-spacing:.07em;color:#5F7081;text-transform:uppercase;}
.qm-number {font-size:25px;line-height:1.14;font-weight:800;letter-spacing:-.04em;margin:4px 0;color:var(--qm-ink);}
.qm-help {font-size:10px;color:#61748A;}
.qm-queue-flow {display:flex;align-items:center;justify-content:space-between;gap:16px;min-height:74px;margin-top:6px;}
.qm-dots {display:flex;flex-wrap:wrap;align-content:flex-start;gap:6px;width:180px;min-height:55px;max-height:98px;overflow:hidden;}
.qm-person {width:11px;height:11px;border-radius:50%;background:var(--qm-navy);transition:transform .25s ease,opacity .25s ease;}
.qm-person:nth-child(5n){background:#2F638F}.qm-served{background:var(--qm-coral);}
.qm-service-stage {display:flex;align-items:center;gap:8px;color:var(--qm-coral);font-size:21px;font-weight:800;padding-right:22px;}
.qm-service-stage small {display:block;color:var(--qm-muted);font-size:9px;font-weight:800;letter-spacing:.04em;}
.qm-service-dots {display:flex;gap:5px;}
.qm-service-dots i {width:12px;height:12px;background:var(--qm-coral);border-radius:50%;display:block;}
.qm-counter-list {display:flex;gap:6px;flex-wrap:wrap;margin-top:10px;}
.qm-counter {background:#EAF7F0;border:1px solid #BCE7D0;color:#1C8054;border-radius:8px;
  padding:5px 8px;font-size:10px;font-weight:700;}
.qm-visual-key {display:flex;gap:14px;font-size:10px;color:var(--qm-muted);margin-top:8px;}
.qm-action-title{font-size:17px;font-weight:800;line-height:1.24;margin:8px 0;color:var(--qm-ink);}
.qm-action-meta{font-size:11px;color:#52677D;}
.qm-impact {display:flex;gap:8px;margin-top:9px;}
.qm-impact>div {flex:1;background:rgba(255,255,255,.72);border-radius:9px;padding:7px 9px;}
.qm-impact b{display:block;font-size:17px;color:var(--qm-navy);}
.qm-timeline {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:7px;}
.qm-time-node {padding:10px 11px;border-radius:11px;background:#fff;border:1px solid var(--qm-border);min-width:0;}
.qm-time-node.exceeded {border-color:#F1C37A;background:#FFF6E5;}
.qm-time-node strong {font-size:21px;letter-spacing:-.035em;color:var(--qm-ink);}
.qm-time-node small {display:block;color:var(--qm-muted);font-size:10px;}
.qm-badge{display:inline-block;padding:3px 7px;border-radius:99px;font-size:9px;font-weight:800;letter-spacing:.04em;}
.qm-badge.normal{color:#1E895A;background:#EAF7F0}
.qm-badge.warning{color:#9B670C;background:#FFF0CB}
.qm-badge.critical{color:#B93636;background:#FDECEC}
.qm-badge.info{color:#2463B1;background:#EAF2FF}
.qm-alert{padding:8px 0;border-bottom:1px solid var(--qm-border);animation:qm-reveal .3s ease-out;}
@keyframes qm-reveal{from{opacity:.5;transform:translateY(3px)}to{opacity:1;transform:translateY(0)}}
.qm-alert:last-child {border-bottom:0;}
.qm-alert b{font-size:11px;margin-left:5px;}
.qm-alert p{font-size:10px;color:var(--qm-muted);margin:3px 0 0;}
.qm-whatif{border:1px solid var(--qm-border);border-radius:12px;padding:11px;background:#fff;min-height:109px;}
.qm-whatif.current{border-color:#F0A99D;background:#FFF7F4;box-shadow:0 0 0 2px rgba(243,111,93,.12)}
.qm-whatif.recommended{border-color:#93D6B0;background:#F2FBF5}
.qm-whatif .big{font-size:24px;font-weight:800;letter-spacing:-.04em;}
.qm-mini-bar{height:7px;border-radius:9px;background:#E7EBEE;overflow:hidden;}
.qm-mini-bar i{display:block;height:100%;background:var(--qm-coral);border-radius:9px;transition:width .3s ease;}
.qm-flow {display:flex;justify-content:space-between;gap:6px;font-size:10px;font-weight:700;text-align:center;}
.qm-flow span {background:#fff;border:1px solid var(--qm-border);border-radius:8px;padding:6px;flex:1;}
.qm-event {display:flex;gap:9px;padding:6px 0;border-bottom:1px solid var(--qm-border);font-size:10px;}
.qm-event time {color:var(--qm-coral);font-weight:800;min-width:30px;}
.qm-muted{color:var(--qm-muted);font-size:11px;}
@media(max-width:1050px){.qm-kpis{grid-template-columns:repeat(3,minmax(0,1fr));}.qm-nav-status{display:none;}}
@media(max-width:650px){.block-container{padding:.6rem .7rem 1.5rem}.qm-nav{gap:7px}.qm-links a{font-size:10px;padding:5px}.qm-kpis{grid-template-columns:repeat(2,minmax(0,1fr));}.qm-timeline{grid-template-columns:repeat(2,minmax(0,1fr));}.qm-title{font-size:22px;}.qm-queue-flow{flex-wrap:wrap}.qm-service-stage{padding-right:0}.qm-visual-key{flex-wrap:wrap;}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important;scroll-behavior:auto!important;}}
</style>
""",
        unsafe_allow_html=True,
    )
