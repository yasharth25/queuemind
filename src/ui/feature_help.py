"""Short, accessible explanations for the main QueueMind features."""

import streamlit as st


FEATURE_HELP = {
    "Home": (
        "A quick view of the queue and the next useful action.",
        "You can start with a video or explore the demo without learning the technical setup.",
        "Upload a canteen clip, or try a simulated lunch rush.",
    ),
    "Upload Video": (
        "QueueMind analyzes a queue recording to count people in a selected area.",
        "You do not have to count everyone manually.",
        "Upload a short queue video, check the queue area, then analyze it.",
    ),
    "Live Demo": (
        "A synthetic queue you can control in real time.",
        "You can see how crowd size and open counters change waiting.",
        "Increase the crowd and open another counter to compare outcomes.",
    ),
    "Predictions": (
        "QueueMind estimates how the queue may change over the next 30 minutes.",
        "You can prepare before the queue gets too long.",
        "A simulated queue of 24 may grow to 34 in 20 minutes.",
    ),
    "Best Counters": (
        "Find the smallest tested number of service counters that meets the wait target.",
        "It can reduce waiting without using more staff than necessary.",
        "Opening one extra counter may reduce the estimated wait.",
    ),
    "Results": (
        "See measured evaluation results and the assumptions behind them.",
        "You can judge what has been tested and what still needs real data.",
        "Compare the model with a simple baseline on held-out synthetic data.",
    ),
}


def feature_heading(name: str, description: str = "") -> None:
    """Native popover supports click/tap, keyboard focus, and outside dismissal."""
    title, help_column = st.columns([12, 1], vertical_alignment="center")
    title.markdown(f"### {name}")
    with help_column:
        with st.popover("ⓘ", help=f"About {name}"):
            what, why, example = FEATURE_HELP[name]
            st.markdown(f"**What is this?**  \n{what}\n\n**Why is it useful?**  \n{why}\n\n**Example**  \n{example}")
    if description:
        st.caption(description)
