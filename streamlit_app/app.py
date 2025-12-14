"""Main Streamlit application entry point for FGEM."""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from streamlit_app.core.config_manager import ConfigManager
from streamlit_app.core.defaults import get_defaults

# Page configuration
st.set_page_config(
    page_title="FGEM - Flexible Geothermal Economics Model",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "config" not in st.session_state:
    st.session_state.config = get_defaults().to_dict()
if "config_manager" not in st.session_state:
    st.session_state.config_manager = ConfigManager()

# Main page content
st.title("🌋 FGEM - Flexible Geothermal Economics Model")
st.markdown("""
**FGEM** (/if'gem/) is an open-source Python library for evaluating lifecycle techno-economics of 
baseload and flexible geothermal energy projects.

This application provides a user-friendly interface to configure, run, and analyze geothermal energy 
project simulations.
""")

# Sidebar
with st.sidebar:
    # Try to load logo if it exists
    logo_path = Path("docs/source/_static/fgem_logo.png")
    if logo_path.exists():
        try:
            st.image(str(logo_path), use_column_width=True)
        except TypeError:
            # Fallback for newer Streamlit versions
            st.image(str(logo_path), width=None)
    else:
        st.markdown("### 🌋 FGEM")
    
    st.markdown("### Navigation")
    st.page_link("app.py", label="🏠 Home", icon="🏠")
    st.page_link("pages/1_Configuration.py", label="⚙️ Configuration", icon="⚙️")
    st.page_link("pages/2_Simulation.py", label="▶️ Simulation", icon="▶️")
    st.page_link("pages/3_Results.py", label="📊 Results", icon="📊")
    st.page_link("pages/4_Sensitivity_Analysis.py", label="🧪 Sensitivity", icon="🧪")
    st.page_link("pages/5_Sensitivity_Results.py", label="📉 Sensitivity Results", icon="📉")
    
    st.markdown("---")
    st.markdown("### Quick Start")
    st.markdown("""
    1. **Configure** your project parameters
    2. **Run** the simulation
    3. **View** results and visualizations
    """)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    For more information, visit:
    - [Documentation](https://fgem.readthedocs.io)
    - [GitHub Repository](https://github.com/aljubrmj/FGEM)
    """)

# Main content sections
st.header("🚀 Getting Started")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### ⚙️ Step 1: Configuration
    
    Configure your geothermal project:
    - Economics parameters
    - Reservoir properties
    - Power plant specifications
    - Market settings
    - Storage options
    """)
    if st.button("Go to Configuration", type="primary", use_container_width=True):
        st.switch_page("pages/1_Configuration.py")

with col2:
    st.markdown("""
    ### ▶️ Step 2: Simulation
    
    Run your simulation:
    - Initialize the model
    - Execute simulation
    - Monitor progress
    """)
    if st.button("Go to Simulation", type="primary", use_container_width=True):
        st.switch_page("pages/2_Simulation.py")

with col3:
    st.markdown("""
    ### 📊 Step 3: Results
    
    Analyze results:
    - Key metrics (NPV, LCOE, ROI)
    - Interactive visualizations
    - Data export
    """)
    if st.button("Go to Results", type="primary", use_container_width=True):
        st.switch_page("pages/3_Results.py")

st.markdown("---")

# Features
st.header("✨ Features")

feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.markdown("""
    - **Comprehensive Configuration**
      - All parameters configurable via UI
      - JSON import/export support
      - Real-time validation
    
    - **Multiple Reservoir Models**
      - Diffusion-convection
      - Energy decline
      - U-loop systems
      - Coaxial systems
    """)

with feature_col2:
    st.markdown("""
    - **Flexible Power Plants**
      - Binary (ORC) cycles
      - Flash systems
      - GEOPHIRES integration
    
    - **Storage Options**
      - Thermal Energy Storage (TES)
      - Lithium-ion batteries
      - Hybrid systems
    """)

st.markdown("---")

# Current configuration summary
st.header("📋 Current Configuration Summary")

if st.session_state.config:
    with st.expander("View Current Configuration"):
        st.json(st.session_state.config)
        
        # Validate
        config_manager = st.session_state.config_manager
        is_valid, errors = config_manager.validate_config(st.session_state.config)
        
        if is_valid:
            st.success("✅ Configuration is valid")
        else:
            st.warning("⚠️ Configuration has issues:")
            for error in errors:
                st.error(f"  • {error}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>FGEM - Flexible Geothermal Economics Model</p>
    <p>For support, contact: aljubrmj@stanford.edu</p>
</div>
""", unsafe_allow_html=True)

