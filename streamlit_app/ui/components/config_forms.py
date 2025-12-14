"""Configuration form components for Streamlit."""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from streamlit_app.core.defaults import get_defaults


def _get_float_value(config: Dict[str, Any], key: str, defaults: Dict[str, Any], fallback: float) -> float:
    """Safely get float value from config or defaults, handling None values.
    
    Args:
        config: Configuration dictionary.
        key: Key to look up.
        defaults: Defaults dictionary.
        fallback: Fallback value if not found or None.
        
    Returns:
        Float value.
    """
    value = config.get(key)
    if value is None:
        value = defaults.get(key)
    if value is None:
        value = fallback
    return float(value)


def _get_int_value(config: Dict[str, Any], key: str, defaults: Dict[str, Any], fallback: int) -> int:
    """Safely get int value from config or defaults, handling None values.
    
    Args:
        config: Configuration dictionary.
        key: Key to look up.
        defaults: Defaults dictionary.
        fallback: Fallback value if not found or None.
        
    Returns:
        Integer value.
    """
    value = config.get(key)
    if value is None:
        value = defaults.get(key)
    if value is None:
        value = fallback
    return int(value)


def render_economics_form(config: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Render economics configuration form.
    
    Args:
        config: Current configuration dictionary.
        defaults: Default values dictionary.
        
    Returns:
        Updated configuration dictionary.
    """
    if defaults is None:
        defaults = get_defaults().to_dict()
    
    st.subheader("Economics Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        config["L"] = st.number_input(
            "Project Lifetime (years)",
            min_value=1,
            max_value=100,
            value=_get_int_value(config, "L", defaults, 30),
            help="Project lifetime in years"
        )
        
        config["d"] = st.number_input(
            "Discount Rate",
            min_value=0.0,
            max_value=1.0,
            value=_get_float_value(config, "d", defaults, 0.07),
            step=0.01,
            format="%.3f",
            help="Discount rate (0-1)"
        )
        
        config["itc"] = st.number_input(
            "Investment Tax Credit",
            min_value=0.0,
            max_value=1.0,
            value=_get_float_value(config, "itc", defaults, 0.0),
            step=0.01,
            format="%.2f",
            help="Investment tax credit fraction (0-1)"
        )
    
    with col2:
        config["inflation"] = st.number_input(
            "Inflation Rate",
            min_value=0.0,
            max_value=0.2,
            value=_get_float_value(config, "inflation", defaults, 0.02),
            step=0.01,
            format="%.3f"
        )
        
        config["contingency"] = st.number_input(
            "Contingency Factor",
            min_value=0.0,
            max_value=1.0,
            value=_get_float_value(config, "contingency", defaults, 0.15),
            step=0.01,
            format="%.2f"
        )
        
        config["powerplant_interconnection_cost"] = st.number_input(
            "Interconnection Cost (USD/kW)",
            min_value=0.0,
            value=_get_float_value(config, "powerplant_interconnection_cost", defaults, 130.0),
            step=10.0
        )

        config["drilling_cost"] = st.number_input(
            "Drilling Cost (MM$ per well)",
            min_value=0.0,
            value=_get_float_value(config, "drilling_cost", defaults, 10.0),
            step=0.5,
            format="%.2f",
            help=(
                "Drilling cost applies to ALL reservoir types (diffusion_convection, uloop, coaxial, etc.). "
                "User input in MM$ per well (default 10.0). "
                "Internally FGEM converts this to USD/m using well depth + laterals. "
                "The conversion accounts for different well geometries (e.g., uloop uses half_lateral_length). "
                "Backward compatibility: if you enter a value > 200, it is treated as USD/m directly. "
                "Set to 0.0 for no drilling cost."
            ),
        )
    
    return config


def render_upstream_form(config: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Render upstream (reservoir + well) configuration form.
    
    Args:
        config: Current configuration dictionary.
        defaults: Default values dictionary.
        
    Returns:
        Updated configuration dictionary.
    """
    if defaults is None:
        defaults = get_defaults().to_dict()
    
    st.subheader("Upstream Parameters")
    
    # Reservoir type
    reservoir_types = ["diffusion_convection", "energy_decline", "uloop", "coaxial", "percentage", "tabular"]
    config["reservoir_type"] = st.selectbox(
        "Reservoir Type",
        options=reservoir_types,
        index=reservoir_types.index(config.get("reservoir_type", defaults.get("reservoir_type", "diffusion_convection"))) if config.get("reservoir_type") in reservoir_types else 0
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Reservoir Properties**")
        config["Tres_init"] = st.number_input(
            "Initial Reservoir Temperature (°C)",
            min_value=0.0,
            max_value=500.0,
            value=_get_float_value(config, "Tres_init", defaults, 225.0),
            step=5.0
        )
        
        config["Pres_init"] = st.number_input(
            "Initial Reservoir Pressure (bar)",
            min_value=0.0,
            value=_get_float_value(config, "Pres_init", defaults, 40.0),
            step=5.0
        )
        
        config["V_res"] = st.number_input(
            "Reservoir Volume (km³)",
            min_value=0.0,
            value=_get_float_value(config, "V_res", defaults, 5.0),
            step=1.0
        )
        
        config["phi_res"] = st.number_input(
            "Porosity",
            min_value=0.0,
            max_value=1.0,
            value=_get_float_value(config, "phi_res", defaults, 0.1),
            step=0.01,
            format="%.2f"
        )
        
        config["geothermal_gradient"] = st.number_input(
            "Geothermal Gradient (°C/km)",
            min_value=0.0,
            value=_get_float_value(config, "geothermal_gradient", defaults, 35.0),
            step=5.0
        )
    
    with col2:
        st.markdown("**Well Properties**")
        config["well_tvd"] = st.number_input(
            "Well TVD (meters)",
            min_value=0.0,
            value=_get_float_value(config, "well_tvd", defaults, 3000.0),
            step=100.0
        )
        
        config["num_prd"] = st.number_input(
            "Number of Production Wells",
            min_value=0,
            value=_get_int_value(config, "num_prd", defaults, 4)
        )
        
        config["inj_prd_ratio"] = st.number_input(
            "Injection/Production Ratio",
            min_value=0.0,
            max_value=2.0,
            value=_get_float_value(config, "inj_prd_ratio", defaults, 1.0),
            step=0.1,
            format="%.1f"
        )
        
        config["prd_well_diam"] = st.number_input(
            "Production Well Diameter (m)",
            min_value=0.0,
            value=_get_float_value(config, "prd_well_diam", defaults, 0.3115),
            step=0.01,
            format="%.4f"
        )
        
        config["inj_well_diam"] = st.number_input(
            "Injection Well Diameter (m)",
            min_value=0.0,
            value=_get_float_value(config, "inj_well_diam", defaults, 0.3115),
            step=0.01,
            format="%.4f"
        )
        
        config["PI"] = st.number_input(
            "Productivity Index (l/s/bar)",
            min_value=0.0,
            value=_get_float_value(config, "PI", defaults, 10.0),
            step=1.0
        )
        
        config["II"] = st.number_input(
            "Injectivity Index (l/s/bar)",
            min_value=0.0,
            value=_get_float_value(config, "II", defaults, 10.0),
            step=1.0
        )
        
        config["m_prd"] = st.number_input(
            "Production Mass Flow Rate (kg/s)",
            min_value=0.0,
            value=_get_float_value(config, "m_prd", defaults, 100.0),
            step=5.0
        )
        
        config["lateral_length"] = st.number_input(
            "Lateral Length (meters)",
            min_value=0.0,
            value=_get_float_value(config, "lateral_length", defaults, 0.0),
            step=100.0
        )
        
        config["numberoflaterals"] = st.number_input(
            "Number of Laterals",
            min_value=1,
            value=_get_int_value(config, "numberoflaterals", defaults, 1)
        )
        
        config["lateral_diam"] = st.number_input(
            "Lateral Diameter (m)",
            min_value=0.0,
            value=_get_float_value(config, "lateral_diam", defaults, 0.3115),
            step=0.01,
            format="%.4f"
        )
        
        config["lateral_spacing"] = st.number_input(
            "Lateral Spacing (meters)",
            min_value=0.0,
            value=_get_float_value(config, "lateral_spacing", defaults, 100.0),
            step=10.0
        )
        
        config["dx"] = st.number_input(
            "Grid Spacing dx (meters) - Optional",
            min_value=0.0,
            value=_get_float_value(config, "dx", defaults, 0.0) if config.get("dx") is not None else None,
            step=50.0,
            help="Grid spacing for numerical simulation. Leave as 0 for default."
        )
        if config.get("dx") == 0.0:
            config["dx"] = None
    
    # Reservoir file upload option
    st.markdown("**Reservoir Data File (Optional)**")
    reservoir_file = st.text_input(
        "Reservoir Filename",
        value=config.get("reservoir_filename", "") or "",
        help="Optional CSV file with reservoir data. Leave empty to use analytical models."
    )
    if reservoir_file:
        config["reservoir_filename"] = reservoir_file
    
    # Simulator Settings (Advanced)
    with st.expander("🔧 Advanced Simulator Settings", expanded=False):
        config = render_simulator_settings_form(config, defaults, reservoir_type=config.get("reservoir_type", "diffusion_convection"))
    
    return config


def render_simulator_settings_form(config: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None, 
                                   reservoir_type: str = "diffusion_convection") -> Dict[str, Any]:
    """Render simulator settings configuration form.
    
    Args:
        config: Current configuration dictionary.
        defaults: Default values dictionary.
        reservoir_type: Current reservoir type to show relevant settings.
        
    Returns:
        Updated configuration dictionary.
    """
    if defaults is None:
        defaults = get_defaults().to_dict()
    
    st.markdown("**General Settings**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        config["ramey"] = st.checkbox(
            "Enable Ramey Model",
            value=config.get("ramey", defaults.get("ramey", True)),
            help="Use Ramey's model for wellbore heat loss/gain"
        )
        
        config["pumping"] = st.checkbox(
            "Account for Pumping Losses",
            value=config.get("pumping", defaults.get("pumping", True)),
            help="Account for parasitic losses due to pumping"
        )
    
    with col2:
        config["timestep_hours"] = st.number_input(
            "Simulation Timestep (hours)",
            min_value=0.1,
            max_value=24.0,
            value=_get_float_value(config, "timestep_hours", defaults, 1.0),
            step=0.1,
            help="Simulation timestep size"
        )
        
        config["impedance"] = st.number_input(
            "Reservoir Impedance",
            min_value=0.0,
            value=_get_float_value(config, "impedance", defaults, 0.1),
            step=0.01,
            help="Reservoir pressure losses"
        )
    
    with col3:
        config["N_ramey_mv_avg"] = st.number_input(
            "Ramey Averaging Timesteps",
            min_value=1,
            value=_get_int_value(config, "N_ramey_mv_avg", defaults, 168),
            help="Timesteps for averaging Ramey's f-function"
        )
    
    st.markdown("**Rock Properties**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        config["krock"] = st.number_input(
            "Rock Thermal Conductivity (W/m-K)",
            min_value=0.0,
            value=_get_float_value(config, "krock", defaults, 3.0),
            step=0.5,
            help="General rock thermal conductivity"
        )
    
    with col2:
        config["rhorock"] = st.number_input(
            "Rock Density (kg/m³)",
            min_value=0.0,
            value=_get_float_value(config, "rhorock", defaults, 2700.0),
            step=100.0
        )
    
    with col3:
        config["cprock"] = st.number_input(
            "Rock Heat Capacity (J/kg-K)",
            min_value=0.0,
            value=_get_float_value(config, "cprock", defaults, 1000.0),
            step=100.0
        )
    
    with col4:
        config["krock_wellbore"] = st.number_input(
            "Wellbore Thermal Conductivity (W/m-K)",
            min_value=0.0,
            value=_get_float_value(config, "krock_wellbore", defaults, 3.0),
            step=0.5
        )
    
    # Reservoir Simulator Settings
    st.markdown("**Reservoir Simulator Settings**")
    col1, col2 = st.columns(2)
    
    with col1:
        # Handle nested reservoir_simulator_settings
        rss = config.get("reservoir_simulator_settings", defaults.get("reservoir_simulator_settings", {}))
        if not isinstance(rss, dict):
            rss = {}
        
        fast_mode = st.checkbox(
            "Fast Mode",
            value=rss.get("fast_mode", False),
            help="Reduce computational requirements in exchange for accuracy"
        )
        
        config["reservoir_simulator_settings"] = rss.copy()
        config["reservoir_simulator_settings"]["fast_mode"] = fast_mode
        
        if fast_mode:
            period_days = rss.get("period", 31536000) / (3600 * 24)  # Convert seconds to days
            period_days = st.number_input(
                "Update Period (days)",
                min_value=1.0,
                value=period_days,
                step=1.0,
                help="Time period before reservoir state is updated"
            )
            config["reservoir_simulator_settings"]["period"] = int(period_days * 3600 * 24)
    
    with col2:
        accuracy = st.selectbox(
            "Accuracy Level",
            options=[1, 2, 3, 4, 5, 6],
            index=rss.get("accuracy", 1) - 1 if isinstance(rss.get("accuracy"), int) else 0,
            help="Numerical accuracy level (1=fastest, 6=most accurate)"
        )
        config["reservoir_simulator_settings"]["accuracy"] = accuracy
        
        dynamic_props = st.checkbox(
            "Dynamic Fluid Properties",
            value=rss.get("DynamicFluidProperties", True),
            help="Update geofluid properties using steam tables"
        )
        config["reservoir_simulator_settings"]["DynamicFluidProperties"] = dynamic_props
    
    # ULoop-specific settings
    if reservoir_type == "uloop":
        st.markdown("**ULoop-Specific Settings**")
        col1, col2 = st.columns(2)
        
        with col1:
            config["k_m"] = st.number_input(
                "ULoop Rock Thermal Conductivity (W/m-K)",
                min_value=0.0,
                value=_get_float_value(config, "k_m", defaults, 2.83),
                step=0.1
            )
            
            config["rho_m"] = st.number_input(
                "ULoop Rock Density (kg/m³)",
                min_value=0.0,
                value=_get_float_value(config, "rho_m", defaults, 2875.0),
                step=50.0
            )
            
            config["c_m"] = st.number_input(
                "ULoop Rock Heat Capacity (J/kg-K)",
                min_value=0.0,
                value=_get_float_value(config, "c_m", defaults, 825.0),
                step=50.0
            )
            
            config["k_f"] = st.number_input(
                "Fluid Thermal Conductivity (W/m-K)",
                min_value=0.0,
                value=_get_float_value(config, "k_f", defaults, 0.68),
                step=0.01
            )
        
        with col2:
            config["mu_f"] = st.number_input(
                "Fluid Viscosity (m²/s)",
                min_value=0.0,
                value=_get_float_value(config, "mu_f", defaults, 6e-4),
                step=1e-5,
                format="%.6f"
            )
            
            config["cp_f"] = st.number_input(
                "Fluid Heat Capacity (J/kg-K)",
                min_value=0.0,
                value=_get_float_value(config, "cp_f", defaults, 4200.0),
                step=100.0
            )
            
            config["rho_f"] = st.number_input(
                "Fluid Density (kg/m³)",
                min_value=0.0,
                value=_get_float_value(config, "rho_f", defaults, 1000.0),
                step=10.0
            )
            
            config["fullyimplicit"] = st.selectbox(
                "Solver Method",
                options=[0, 1],
                index=config.get("fullyimplicit", defaults.get("fullyimplicit", 1)),
                help="Euler's method solver option"
            )
            
            config["FMM"] = st.selectbox(
                "Fast Multi-Pole Method",
                options=[0, 1],
                index=config.get("FMM", defaults.get("FMM", 1)),
                help="Enable fast multi-pole method"
            )
            
            fmm_trigger_days = config.get("FMMtriggertime", defaults.get("FMMtriggertime", 864000.0)) / (3600 * 24)
            fmm_trigger_days = st.number_input(
                "FMM Trigger Time (days)",
                min_value=0.0,
                value=fmm_trigger_days,
                step=1.0
            )
            config["FMMtriggertime"] = fmm_trigger_days * 3600 * 24
    
    # Coaxial-specific settings
    elif reservoir_type == "coaxial":
        st.markdown("**Coaxial-Specific Settings**")
        col1, col2 = st.columns(2)
        
        with col1:
            config["k_m"] = st.number_input(
                "Coaxial Rock Thermal Conductivity (W/m-K)",
                min_value=0.0,
                value=_get_float_value(config, "k_m", defaults, 2.3),
                step=0.1
            )
            
            config["rho_m"] = st.number_input(
                "Coaxial Rock Density (kg/m³)",
                min_value=0.0,
                value=_get_float_value(config, "rho_m", defaults, 2875.0),
                step=50.0
            )
            
            config["c_m"] = st.number_input(
                "Coaxial Rock Heat Capacity (J/kg-K)",
                min_value=0.0,
                value=_get_float_value(config, "c_m", defaults, 825.0),
                step=50.0
            )
        
        with col2:
            config["k_f"] = st.number_input(
                "Coaxial Fluid Thermal Conductivity (W/m-K)",
                min_value=0.0,
                value=_get_float_value(config, "k_f", defaults, 0.6),
                step=0.01
            )
            
            config["mu_f"] = st.number_input(
                "Coaxial Fluid Viscosity (m²/s)",
                min_value=0.0,
                value=_get_float_value(config, "mu_f", defaults, 6e-4),
                step=1e-5,
                format="%.6f"
            )
            
            config["cp_f"] = st.number_input(
                "Coaxial Fluid Heat Capacity (J/kg-K)",
                min_value=0.0,
                value=_get_float_value(config, "cp_f", defaults, 4184.0),
                step=100.0
            )
            
            config["rho_f"] = st.number_input(
                "Coaxial Fluid Density (kg/m³)",
                min_value=0.0,
                value=_get_float_value(config, "rho_f", defaults, 990.0),
                step=10.0
            )
            
            config["FMM"] = st.selectbox(
                "Fast Multi-Pole Method",
                options=[0, 1],
                index=config.get("FMM", defaults.get("FMM", 1)),
                help="Enable fast multi-pole method"
            )
            
            fmm_trigger_days = config.get("FMMtriggertime", defaults.get("FMMtriggertime", 864000.0)) / (3600 * 24)
            fmm_trigger_days = st.number_input(
                "FMM Trigger Time (days)",
                min_value=0.0,
                value=fmm_trigger_days,
                step=1.0
            )
            config["FMMtriggertime"] = fmm_trigger_days * 3600 * 24
    
    # DiffusionConvection-specific overrides
    elif reservoir_type == "diffusion_convection":
        st.markdown("**Diffusion-Convection Specific Overrides**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            config["krock_diffusion"] = st.number_input(
                "Diffusion Rock Thermal Conductivity (W/m-K)",
                min_value=0.0,
                value=_get_float_value(config, "krock_diffusion", defaults, 30.0),
                step=1.0,
                help="Overrides general krock for diffusion_convection"
            )
        
        with col2:
            config["rhorock_diffusion"] = st.number_input(
                "Diffusion Rock Density (kg/m³)",
                min_value=0.0,
                value=_get_float_value(config, "rhorock_diffusion", defaults, 2600.0),
                step=50.0
            )
        
        with col3:
            config["cprock_diffusion"] = st.number_input(
                "Diffusion Rock Heat Capacity (J/kg-K)",
                min_value=0.0,
                value=_get_float_value(config, "cprock_diffusion", defaults, 1100.0),
                step=50.0
            )
    
    return config


def render_downstream_form(config: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Render downstream (power plant) configuration form.
    
    Args:
        config: Current configuration dictionary.
        defaults: Default values dictionary.
        
    Returns:
        Updated configuration dictionary.
    """
    if defaults is None:
        defaults = get_defaults().to_dict()
    
    st.subheader("Downstream Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        powerplant_types = ["Binary", "Flash", "ORC", "GEOPHIRES", "HighEnthalpyCLGWGPowerPlant"]
        current_pp_type = config.get("powerplant_type", defaults.get("powerplant_type", "Binary"))
        pp_type_index = powerplant_types.index(current_pp_type) if current_pp_type in powerplant_types else 0
        
        config["powerplant_type"] = st.selectbox(
            "Power Plant Type",
            options=powerplant_types,
            index=pp_type_index
        )
        
        powerplant_capacity_val = config.get("powerplant_capacity")
        if powerplant_capacity_val is None:
            powerplant_capacity_val = defaults.get("powerplant_capacity")
        config["powerplant_capacity"] = st.number_input(
            "Power Plant Capacity (MWe)",
            min_value=0.0,
            value=float(powerplant_capacity_val) if powerplant_capacity_val is not None else None,
            step=10.0,
            help="Leave empty to auto-calculate"
        )
        
        config["cf"] = st.number_input(
            "Capacity Factor",
            min_value=0.0,
            max_value=1.0,
            value=_get_float_value(config, "cf", defaults, 1.0),
            step=0.1,
            format="%.2f"
        )
    
    with col2:
        config["pipinglength"] = st.number_input(
            "Piping Length (km)",
            min_value=0.0,
            value=_get_float_value(config, "pipinglength", defaults, 5.0),
            step=1.0
        )
        
        config["bypass"] = st.checkbox(
            "Enable Bypass",
            value=config.get("bypass", defaults.get("bypass", False))
        )
    
    return config


def render_market_form(config: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Render market configuration form.
    
    Args:
        config: Current configuration dictionary.
        defaults: Default values dictionary.
        
    Returns:
        Updated configuration dictionary.
    """
    if defaults is None:
        defaults = get_defaults().to_dict()
    
    st.subheader("Market Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        config["energy_price"] = st.number_input(
            "Energy Price ($/MWh)",
            min_value=0.0,
            value=_get_float_value(config, "energy_price", defaults, 40.0),
            step=5.0,
            help="Used if energy_market_filename is not provided"
        )
        
        config["capacity_price"] = st.number_input(
            "Capacity Price ($/MW-hour)",
            min_value=0.0,
            value=_get_float_value(config, "capacity_price", defaults, 100.0),
            step=10.0
        )
        
        config["recs_price"] = st.number_input(
            "RECs Price ($/MWh)",
            min_value=0.0,
            value=_get_float_value(config, "recs_price", defaults, 10.0),
            step=1.0
        )
    
    with col2:
        config["ppa_price"] = st.number_input(
            "PPA Price ($/MWh)",
            min_value=0.0,
            value=_get_float_value(config, "ppa_price", defaults, 70.0),
            step=5.0
        )
        
        config["ppa_escalaction_rate"] = st.number_input(
            "PPA Escalation Rate",
            min_value=0.0,
            max_value=0.2,
            value=_get_float_value(config, "ppa_escalaction_rate", defaults, 0.02),
            step=0.01,
            format="%.3f"
        )
        
        config["fat_factor"] = st.number_input(
            "Fat Factor",
            min_value=0.0,
            value=_get_float_value(config, "fat_factor", defaults, 1.0),
            step=0.1
        )
    
    return config


def render_storage_form(config: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Render storage (battery + TES) configuration form.
    
    Args:
        config: Current configuration dictionary.
        defaults: Default values dictionary.
        
    Returns:
        Updated configuration dictionary.
    """
    if defaults is None:
        defaults = get_defaults().to_dict()
    
    st.subheader("Storage Parameters")
    
    # Battery
    st.markdown("**Battery Storage**")
    col1, col2 = st.columns(2)
    
    with col1:
        battery_duration = config.get("battery_duration", defaults.get("battery_duration", [0, 0]))
        if not isinstance(battery_duration, list):
            battery_duration = [battery_duration, 0]
        
        config["battery_duration_0"] = st.number_input(
            "Battery Duration Unit 1 (hours)",
            min_value=0.0,
            value=float(battery_duration[0] if len(battery_duration) > 0 else 0.0),
            step=1.0
        )
        
        battery_power = config.get("battery_power_capacity", defaults.get("battery_power_capacity", [0, 0]))
        if not isinstance(battery_power, list):
            battery_power = [battery_power, 0]
        
        config["battery_power_capacity_0"] = st.number_input(
            "Battery Power Capacity Unit 1 (MWe)",
            min_value=0.0,
            value=float(battery_power[0] if len(battery_power) > 0 else 0.0),
            step=10.0
        )
    
    with col2:
        config["battery_duration_1"] = st.number_input(
            "Battery Duration Unit 2 (hours)",
            min_value=0.0,
            value=float(battery_duration[1] if len(battery_duration) > 1 else 0.0),
            step=1.0
        )
        
        config["battery_power_capacity_1"] = st.number_input(
            "Battery Power Capacity Unit 2 (MWe)",
            min_value=0.0,
            value=float(battery_power[1] if len(battery_power) > 1 else 0.0),
            step=10.0
        )
    
    # Convert back to list format
    config["battery_duration"] = [config.pop("battery_duration_0", 0), config.pop("battery_duration_1", 0)]
    config["battery_power_capacity"] = [config.pop("battery_power_capacity_0", 0), config.pop("battery_power_capacity_1", 0)]
    
    # TES
    st.markdown("**Thermal Energy Storage**")
    col1, col2 = st.columns(2)
    
    with col1:
        config["tank_diameter"] = st.number_input(
            "Tank Diameter (m)",
            min_value=0.0,
            value=_get_float_value(config, "tank_diameter", defaults, 0.0),
            step=1.0
        )
    
    with col2:
        config["tank_height"] = st.number_input(
            "Tank Height (m)",
            min_value=0.0,
            value=_get_float_value(config, "tank_height", defaults, 0.0),
            step=1.0
        )
    
    return config

