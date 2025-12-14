"""Progress indicator components for Streamlit."""

import streamlit as st
from typing import Optional, Callable


def render_progress_bar(progress: float, current_step: int, total_steps: int):
    """Render a progress bar.
    
    Args:
        progress: Progress percentage (0-100).
        current_step: Current step number.
        total_steps: Total number of steps.
    """
    st.progress(progress / 100.0)
    st.caption(f"Step {current_step:,} of {total_steps:,} ({progress:.1f}%)")


def render_simulation_status(state: str, error_message: Optional[str] = None):
    """Render simulation status indicator.
    
    Args:
        state: Current simulation state.
        error_message: Error message if state is 'error'.
    """
    status_colors = {
        "idle": "⚪",
        "running": "🟢",
        "paused": "🟡",
        "completed": "✅",
        "error": "❌"
    }
    
    status_labels = {
        "idle": "Ready",
        "running": "Running",
        "paused": "Paused",
        "completed": "Completed",
        "error": "Error"
    }
    
    icon = status_colors.get(state, "⚪")
    label = status_labels.get(state, "Unknown")
    
    st.markdown(f"**Status:** {icon} {label}")
    
    if state == "error" and error_message:
        st.error(f"Error: {error_message}")

