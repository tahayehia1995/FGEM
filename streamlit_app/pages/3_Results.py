"""Results page for FGEM Streamlit app."""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from streamlit_app.core.simulation_engine import SimulationEngine
from streamlit_app.visualization.economics_plots import EconomicsVisualizer
from streamlit_app.visualization.operations_plots import OperationsVisualizer
from streamlit_app.visualization.reservoir_plots import ReservoirVisualizer

# Page configuration
st.set_page_config(page_title="Results - FGEM", page_icon="📊", layout="wide")

st.title("📊 Results")
st.markdown("View simulation results and visualizations.")

# Check if simulation is completed
if "simulation_engine" not in st.session_state or st.session_state.simulation_engine is None:
    st.warning("⚠️ No simulation results found. Please run a simulation first.")
    if st.button("Go to Simulation"):
        st.switch_page("pages/2_Simulation.py")
    st.stop()

engine: SimulationEngine = st.session_state.simulation_engine

# Check if config has changed since simulation was run
if "config" in st.session_state and "config_hash" in st.session_state:
    import hashlib
    current_config = st.session_state.config
    current_config_str = str(sorted(current_config.items()))
    current_config_hash = hashlib.md5(current_config_str.encode()).hexdigest()
    
    if current_config_hash != st.session_state.config_hash:
        st.warning("⚠️ Configuration has changed since this simulation was run. Results may not match current configuration.")
        st.info("💡 Tip: Run a new simulation with the updated configuration to see updated results.")

world = engine.get_results()

if world is None:
    st.warning("⚠️ Simulation not completed. Please run a simulation first.")
    if st.button("Go to Simulation"):
        st.switch_page("pages/2_Simulation.py")
    st.stop()

# Key metrics
st.header("📈 Key Metrics")

# Compute economics if not already computed
if not hasattr(world, "NPV"):
    world.compute_economics(print_outputs=False)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("NPV", f"${world.NPV:.1f} MM", delta=None)
    st.metric("LCOE", f"${world.LCOE:.0f}/MWh", delta=None)

with col2:
    st.metric("ROI", f"{world.ROI:.1f}%", delta=None)
    st.metric("IRR", f"{world.IRR:.2f}%", delta=None)

with col3:
    st.metric("Payback Period", f"{world.PBP:.1f} years", delta=None)
    st.metric("Net Generation", f"{world.NET_GEN/1e6:.1f} TWh", delta=None)

with col4:
    st.metric("Capacity Factor", f"{world.NET_CF*100:.1f}%", delta=None)
    st.metric("Avg Ambient Temp", f"{world.AVG_T_AMB:.1f}°C", delta=None)

# Visualizations
st.header("📊 Visualizations")

viz_tab1, viz_tab2, viz_tab3 = st.tabs([
    "Economics",
    "Operations",
    "Reservoir"
])

with viz_tab1:
    st.subheader("CAPEX and OPEX Breakdown")
    
    plot_type = st.radio("Plot Type", ["Plotly (Interactive)", "Matplotlib"], horizontal=True)
    
    if plot_type == "Plotly (Interactive)":
        fig = EconomicsVisualizer.plot_capex_opex_plotly(world)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = EconomicsVisualizer.plot_capex_opex_matplotlib(world)
        st.pyplot(fig)
    
    with st.expander("📖 How CAPEX and OPEX are Calculated", expanded=False):
        st.markdown("""
        ### Powerplant CAPEX Calculation
        
        Powerplant capital costs are computed using **temperature-dependent correlations** that vary by powerplant type:
        
        - **Input**: Median wellhead temperature from simulation (or design temperature `Tres_pp_design` if not available)
        - **Method**: Each powerplant type (`Binary`, `ORC`, `GEOPHIRES ORC`, `GEOPHIRES Flash`, `Flash`) implements its own `compute_cplant()` correlation
        - **Scaling**: Costs are scaled by capacity using economies of scale factors (e.g., `(capacity/15)^-0.06` for GEOPHIRES ORC)
        - **Minimum**: Enforced minimum cost per kW (`powerplant_usd_per_kw_min`, default 2000 USD/kW)
        - **Allocation**: All powerplant CAPEX is allocated to year 0 (project start)
        
        **Interconnection CAPEX** is computed separately:
        - `interconnection_cost * capacity * 1000 / 1e6` ($MM)
        - Also allocated to year 0 only
        
        ### Powerplant OPEX Calculation
        
        Annual operating costs are computed as:
        
        ```
        Annual OPEX = (powerplant_opex_rate × Cplant) + (powerplant_labor_rate × Clabor)
        ```
        
        Where:
        - `powerplant_opex_rate`: Default 1.5% of CAPEX per year
        - `powerplant_labor_rate`: Default 0.75 (75% of labor correlation)
        - `Clabor`: Labor cost correlation = `max(1.1 × (589 × log(capacity) - 304) / 1000, 0)` ($MM/year)
        
        OPEX is applied for all project years (`L`) and is **only charged when wells are active** (multiplied by `wells_are_active_01`).
        
        ### Other CAPEX Components
        
        Additional capital costs computed separately:
        - **Drilling**: Based on well depth, diameters, laterals, and drilling cost ($/m or MM$/well)
        - **Exploration**: Intercept + slope × producer well cost
        - **Stimulation**: Cost per well × number of wells (if SSR > 0)
        - **Pumps**: Injection and production pumps based on maximum pumping power requirements
        - **Gathering System**: Fixed cost per well
        - **Pipelines**: Cost per km of piping length
        - **TES/Battery**: Storage component costs
        - **Contingency**: Percentage of total CAPEX (default 15%)
        
        All CAPEX components are discounted to present value using the project discount rate (`d`).
        """)
    
    st.subheader("Price Distribution")
    plot_type2 = st.radio("Plot Type", ["Plotly (Interactive)", "Matplotlib"], 
                          horizontal=True, key="price_dist")
    
    if plot_type2 == "Plotly (Interactive)":
        fig2 = EconomicsVisualizer.plot_price_distribution_plotly(world)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        fig2 = EconomicsVisualizer.plot_price_distribution_matplotlib(world)
        st.pyplot(fig2)
    
    st.subheader("Net Cashflow at Different Discount Rates")
    
    st.markdown("""
    View cumulative discounted cashflow over time for multiple discount rates. 
    Payback periods are marked where cumulative cashflow crosses zero.
    """)
    
    # Discount rate selector
    default_rates = [0.03, 0.05, 0.07, 0.10]
    rate_input = st.text_input(
        "Discount rates (comma-separated, e.g., 0.03,0.05,0.07,0.10)",
        value=", ".join([f"{r:.2f}" for r in default_rates]),
        help="Enter discount rates as decimal fractions (e.g., 0.05 for 5%)"
    )
    
    try:
        discount_rates = [float(r.strip()) for r in rate_input.split(",")]
        discount_rates = [d for d in discount_rates if 0 <= d <= 1]  # Validate range
        if not discount_rates:
            discount_rates = default_rates
    except Exception:
        discount_rates = default_rates
        st.warning("Invalid discount rate input. Using defaults.")
    
    plot_type3 = st.radio("Plot Type", ["Plotly (Interactive)", "Matplotlib"], 
                          horizontal=True, key="cashflow_discount")
    
    if plot_type3 == "Plotly (Interactive)":
        fig3 = EconomicsVisualizer.plot_cashflow_multi_discount_plotly(world, discount_rates)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        fig3 = EconomicsVisualizer.plot_cashflow_multi_discount_matplotlib(world, discount_rates)
        st.pyplot(fig3)

with viz_tab2:
    st.subheader("Operational Parameters")
    
    plot_type = st.radio("Plot Type", ["Plotly (Interactive)", "Matplotlib"], 
                         horizontal=True, key="ops_plot")
    
    # Time range selector
    max_steps = len(world.df_records)
    start_step = st.slider("Start Step", 0, max_steps-1, int(0.01*max_steps))
    end_step = st.slider("End Step", 0, max_steps-1, max_steps-1)
    span = range(start_step, end_step)
    
    if plot_type == "Plotly (Interactive)":
        fig = OperationsVisualizer.plot_operations_plotly(world, span=span)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = OperationsVisualizer.plot_operations_matplotlib(world, span=span)
        st.pyplot(fig)

with viz_tab3:
    st.subheader("Reservoir Visualization")
    
    if hasattr(world, "reservoirs") and len(world.reservoirs) > 0:
        reservoir = world.reservoirs[0]
        
        if hasattr(reservoir, "zprod"):
            plot_type = st.radio("Plot Type", ["Plotly (Interactive)", "Matplotlib"], 
                               horizontal=True, key="res_plot")
            
            if plot_type == "Plotly (Interactive)":
                fig = ReservoirVisualizer.plot_doublet_plotly(reservoir)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = ReservoirVisualizer.plot_doublet_matplotlib(reservoir)
                st.pyplot(fig)
        else:
            st.info("Reservoir visualization requires well dimensions to be configured.")
    else:
        st.info("No reservoir data available for visualization.")

# Data export
st.header("💾 Export Data")

col1, col2 = st.columns(2)

with col1:
    if st.button("Export Records to CSV"):
        csv = world.df_records.to_csv(index=True)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="fgem_records.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Export Annual Summary to CSV"):
        if hasattr(world, "df_annual_nominal"):
            csv = world.df_annual_nominal.to_csv(index=True)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="fgem_annual_summary.csv",
                mime="text/csv"
            )

# Navigation
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("◀️ Back to Simulation"):
        st.switch_page("pages/2_Simulation.py")
with col2:
    if st.button("🔄 New Configuration"):
        # Reset session state
        for key in list(st.session_state.keys()):
            if key != "config":
                del st.session_state[key]
        st.switch_page("pages/1_Configuration.py")

