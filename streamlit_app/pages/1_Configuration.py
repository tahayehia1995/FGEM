"""Configuration page for FGEM Streamlit app."""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from streamlit_app.core.config_manager import ConfigManager
from streamlit_app.core.defaults import get_defaults
from streamlit_app.ui.components.file_upload import upload_config_file, download_config_file
from streamlit_app.ui.components.config_forms import (
    render_economics_form,
    render_upstream_form,
    render_downstream_form,
    render_market_form,
    render_storage_form
)

# Page configuration
st.set_page_config(page_title="Configuration - FGEM", page_icon="⚙️", layout="wide")

st.title("⚙️ Configuration")
st.markdown("Configure your geothermal energy project parameters.")

# Initialize session state
if "config" not in st.session_state:
    st.session_state.config = get_defaults().to_dict()
if "config_manager" not in st.session_state:
    st.session_state.config_manager = ConfigManager()

config_manager = st.session_state.config_manager
config = st.session_state.config.copy()

# File upload section
st.header("📁 Load Configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Config File")
    uploaded_result = upload_config_file()
    if uploaded_result:
        uploaded_config, config_file_path = uploaded_result
        # Merge uploaded config with defaults (pass file path for relative path resolution)
        config = config_manager.merge_config(uploaded_config, config_file_path=config_file_path)
        st.session_state.config = config
        st.session_state.config_file_path = config_file_path
        
        # Update config hash and reset simulation if needed
        import hashlib
        config_str = str(sorted(config.items()))
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        if "config_hash" in st.session_state and st.session_state.config_hash != config_hash:
            # Config changed - reset simulation state
            if "simulation_engine" in st.session_state:
                if st.session_state.simulation_engine is not None:
                    st.session_state.simulation_engine.reset()
                st.session_state.simulation_engine = None
            st.session_state.simulation_running = False
        st.session_state.config_hash = config_hash
        st.success("Configuration loaded from file!")

with col2:
    st.subheader("Load Example Config")
    example_configs = {
        "Example A - Percentage Decline": "examples/configs/exampleA.json",
        "Example B - Percentage Decline with Weather": "examples/configs/exampleB.json",
        "Example C - Percentage Decline": "examples/configs/exampleC.json",
        "Example D - Diffusion Convection": "examples/configs/exampleD.json",
        "Example E - Diffusion Convection": "examples/configs/exampleE.json",
        "Example F - Diffusion Convection": "examples/configs/exampleF.json",
        "Example G - Diffusion Convection with Battery": "examples/configs/exampleG.json",
        "Example H - Diffusion Convection (Reservoir File)": "examples/configs/exampleH.json",
        "Example I - U-Loop": "examples/configs/exampleI.json",
        "Example J - U-Loop (Eavor Design)": "examples/configs/exampleJ.json",
        "Example K - Percentage Decline (Sup3rCC)": "examples/configs/exampleK.json",
        "Example L - Scalable EGS (Diffusion Convection)": "examples/configs/exampleL_Scalable_EGS.json",
        "Example - Coaxial Validation": "examples/configs/example_validate_coaxial.json",
    }
    
    selected_example = st.selectbox(
        "Select an example configuration:",
        options=["None"] + list(example_configs.keys()),
        help="Load a pre-configured example to get started"
    )
    
    if selected_example != "None":
        example_path = example_configs[selected_example]
        if Path(example_path).exists():
            import json
            import hashlib
            with open(example_path, 'r') as f:
                example_config = json.load(f)
            config = config_manager.merge_config(example_config, config_file_path=example_path)
            st.session_state.config = config
            st.session_state.config_file_path = example_path
            
            # Update config hash and reset simulation if needed
            config_str = str(sorted(config.items()))
            config_hash = hashlib.md5(config_str.encode()).hexdigest()
            if "config_hash" in st.session_state and st.session_state.config_hash != config_hash:
                # Config changed - reset simulation state
                if "simulation_engine" in st.session_state:
                    if st.session_state.simulation_engine is not None:
                        st.session_state.simulation_engine.reset()
                    st.session_state.simulation_engine = None
                st.session_state.simulation_running = False
            st.session_state.config_hash = config_hash
            st.success(f"Loaded {selected_example}!")
        else:
            st.error(f"Example file not found: {example_path}")

# Configuration tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Economics",
    "⬆️ Upstream",
    "⬇️ Downstream",
    "📊 Market",
    "🔋 Storage"
])

with tab1:
    config = render_economics_form(config)

with tab2:
    config = render_upstream_form(config)

with tab3:
    config = render_downstream_form(config)

with tab4:
    config = render_market_form(config)

with tab5:
    config = render_storage_form(config)

# Save configuration and update config hash to track changes
# Merge config first to ensure consistent hashing (same as Simulation page)
import hashlib
config_file_path = st.session_state.get("config_file_path")
merged_config = config_manager.merge_config(config, config_file_path=config_file_path)
config_str = str(sorted(merged_config.items()))
config_hash = hashlib.md5(config_str.encode()).hexdigest()

# Check if config has changed
if "config_hash" not in st.session_state:
    st.session_state.config_hash = config_hash
elif st.session_state.config_hash != config_hash:
    # Config has changed - reset simulation state
    if "simulation_engine" in st.session_state:
        if st.session_state.simulation_engine is not None:
            st.session_state.simulation_engine.reset()
        st.session_state.simulation_engine = None
    st.session_state.simulation_running = False
    st.session_state.config_hash = config_hash
    st.info("ℹ️ Configuration updated. The simulation will be reset when you navigate to the Simulation page.")

st.session_state.config = config

# Validation
st.header("✅ Validation")
is_valid, errors = config_manager.validate_config(config)

if is_valid:
    st.success("Configuration is valid!")
else:
    st.error("Configuration has errors:")
    for error in errors:
        st.error(f"  • {error}")

# Export configuration
st.header("💾 Export Configuration")
download_config_file(config, "fgem_config.json")

# Navigation
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("◀️ Back to Home"):
        st.switch_page("app.py")
with col2:
    if is_valid:
        if st.button("Next: Simulation ▶️"):
            st.switch_page("pages/2_Simulation.py")
    else:
        st.button("Next: Simulation ▶️", disabled=True)

