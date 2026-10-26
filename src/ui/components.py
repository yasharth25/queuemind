import streamlit as st
def header(title, subtitle): st.markdown(f'<div class="eyebrow">QUEUE MIND / OPERATIONS</div><div class="page-title">{title}</div><div class="subtle">{subtitle}</div>',unsafe_allow_html=True)
def metric(label,value,detail): st.markdown(f'<div class="card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-detail">{detail}</div></div>',unsafe_allow_html=True)
def action_card(status, message, recommendation): st.markdown(f'<div class="card"><div class="metric-label">ACTION CENTER</div><div class="status {status}">{status.upper()} RISK</div><p>{message}</p><div class="metric-detail">{recommendation}</div></div>',unsafe_allow_html=True)
