"""Simulation page for FGEM Streamlit app."""

import streamlit as st
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from streamlit_app.core.simulation_engine import SimulationEngine
from streamlit_app.core.config_manager import ConfigManager
from streamlit_app.ui.components.progress import render_progress_bar, render_simulation_status

# Page configuration
st.set_page_config(page_title="Simulation - FGEM", page_icon="▶️", layout="wide")

st.title("▶️ Simulation")
st.markdown("Run your geothermal energy project simulation.")

# Initialize session state
if "config" not in st.session_state:
    st.warning("⚠️ No configuration found. Please configure your project first.")
    if st.button("Go to Configuration"):
        st.switch_page("pages/1_Configuration.py")
    st.stop()

if "simulation_engine" not in st.session_state:
    st.session_state.simulation_engine = None
if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False

config = st.session_state.config
config_manager = ConfigManager()

# Check if config has changed since last initialization
# Merge config first to ensure consistent comparison
import hashlib
config_file_path = st.session_state.get("config_file_path")
merged_config = config_manager.merge_config(config, config_file_path=config_file_path)
config_str = str(sorted(merged_config.items()))
current_config_hash = hashlib.md5(config_str.encode()).hexdigest()

if "config_hash" not in st.session_state:
    st.session_state.config_hash = current_config_hash
elif st.session_state.config_hash != current_config_hash:
    # Config has changed - reset simulation engine
    if st.session_state.simulation_engine is not None:
        st.session_state.simulation_engine.reset()
        st.session_state.simulation_engine = None
    st.session_state.simulation_running = False
    st.session_state.config_hash = current_config_hash
    st.warning("⚠️ Configuration has changed. Please reinitialize the simulation.")
    st.info("💡 Tip: Click 'Initialize Simulation' below to start with the new configuration.")

# Configuration summary
st.header("📋 Configuration Summary")
with st.expander("View Configuration"):
    st.json(config)

# Initialize simulation engine
if st.session_state.simulation_engine is None:
    if st.button("Initialize Simulation"):
        with st.spinner("Initializing simulation..."):
            # Get config file path if available (for relative path resolution)
            config_file_path = st.session_state.get("config_file_path")
            if config_file_path:
                # Re-merge config with file path for proper path resolution
                config = config_manager.merge_config(config, config_file_path=config_file_path)
                st.session_state.config = config
            
            engine = SimulationEngine(config, config_manager)
            if engine.initialize():
                st.session_state.simulation_engine = engine
                st.session_state.config_hash = current_config_hash  # Update hash after successful init
                st.success("Simulation initialized successfully!")
                st.rerun()
            else:
                st.error(f"Initialization failed: {engine.error_message}")

# Run simulation
progress_info = {"state": "idle"}  # Default progress info

if st.session_state.simulation_engine is not None:
    engine = st.session_state.simulation_engine
    
    # Verify engine initialized config matches current merged config
    # Compare the merged configs (both should be merged with defaults)
    if engine.initialized_config is not None:
        engine_config_str = str(sorted(engine.initialized_config.items()))
        engine_config_hash = hashlib.md5(engine_config_str.encode()).hexdigest()
        
        if engine_config_hash != current_config_hash:
            # Config mismatch - reset engine
            st.warning("⚠️ Configuration mismatch detected. Resetting simulation engine.")
            engine.reset()
            st.session_state.simulation_engine = None
            st.session_state.simulation_running = False
            st.rerun()
    
    st.header("🚀 Run Simulation")
    
    # Status display
    progress_info = engine.get_progress()
    render_simulation_status(progress_info["state"], progress_info.get("error_message"))
    
    # Progress bar
    if progress_info["state"] == "running":
        render_progress_bar(
            progress_info["progress"],
            progress_info["current_step"],
            progress_info["total_steps"]
        )
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if progress_info["state"] == "idle":
            if st.button("▶️ Start Simulation", type="primary"):
                st.session_state.simulation_running = True
                st.rerun()
    
    with col2:
        if progress_info["state"] == "running":
            if st.button("⏸️ Pause"):
                engine.pause()
                st.rerun()
    
    with col3:
        if progress_info["state"] in ["running", "paused"]:
            if st.button("🛑 Stop"):
                engine.reset()
                st.session_state.simulation_engine = None
                st.session_state.simulation_running = False
                st.rerun()
        elif progress_info["state"] in ["idle", "completed", "error"]:
            if st.button("🔄 Reinitialize"):
                engine.reset()
                st.session_state.simulation_engine = None
                st.session_state.simulation_running = False
                st.rerun()
    
    # Run simulation if started
    if st.session_state.simulation_running and progress_info["state"] == "idle":
        def progress_callback(progress, current_step, total_steps):
            st.session_state.progress = progress
            st.session_state.current_step = current_step
            st.session_state.total_steps = total_steps
        
        with st.spinner("Running simulation..."):
            success = engine.run(progress_callback=progress_callback)
            st.session_state.simulation_running = False
            
            if success:
                st.success("✅ Simulation completed successfully!")
                st.balloons()
            else:
                st.error(f"❌ Simulation failed: {engine.error_message}")
        
        st.rerun()
    
    # Results available
    if progress_info["state"] == "completed":
        st.success("Simulation completed! Proceed to results page.")
        if st.button("View Results ▶️"):
            st.switch_page("pages/3_Results.py")

# Navigation
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("◀️ Back to Configuration"):
        st.switch_page("pages/1_Configuration.py")
with col2:
    if progress_info.get("state") == "completed":
        if st.button("Next: Results ▶️"):
            st.switch_page("pages/3_Results.py")

