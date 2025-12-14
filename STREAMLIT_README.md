# FGEM Streamlit Application Guide

## Overview

The FGEM Streamlit application provides a user-friendly web interface for configuring and running geothermal energy project simulations. The application follows a wizard-style navigation with three main pages: Configuration, Simulation, and Results.

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python run_streamlit.py
   ```
   
   Or directly:
   ```bash
   streamlit run streamlit_app/app.py
   ```

3. **Access the application:**
   - The app will open automatically in your default web browser
   - Default URL: http://localhost:8501

## Project Structure

```
streamlit_app/
├── app.py                    # Main entry point
├── core/                     # Business logic
│   ├── config_manager.py    # Configuration loading, merging, validation
│   ├── defaults.py          # Default parameter values
│   └── simulation_engine.py # Simulation execution wrapper
├── ui/                       # UI components
│   └── components/
│       ├── config_forms.py  # Configuration form widgets
│       ├── file_upload.py   # File upload handlers
│       └── progress.py      # Progress indicators
├── pages/                    # Wizard pages
│   ├── 1_Configuration.py   # Configuration input page
│   ├── 2_Simulation.py      # Simulation execution page
│   └── 3_Results.py         # Results visualization page
├── visualization/            # Plotting functions
│   ├── economics_plots.py   # Economic visualizations
│   ├── operations_plots.py  # Operations visualizations
│   ├── reservoir_plots.py   # Reservoir visualizations
│   └── utils.py             # Plotting utilities
└── fgem/                     # Core simulation module
    ├── world.py             # Main World class
    ├── subsurface.py        # Reservoir models
    ├── powerplant.py        # Power plant models
    ├── markets.py           # Market data handling
    ├── weather.py           # Weather data handling
    ├── storage.py           # Storage systems
    └── utils/                # Utility functions
```

## Usage Guide

### Page 1: Configuration

The Configuration page allows you to set up your geothermal project parameters:

1. **Load Configuration:**
   - Upload a JSON configuration file, or
   - Select an example configuration from the dropdown menu

2. **Configure Parameters:**
   - **Economics**: Project lifetime, discount rate, ITC, inflation, etc.
   - **Upstream**: Reservoir type, well properties, production/injection parameters
     - **Advanced Simulator Settings**: Expand the "Advanced Simulator Settings" section to configure:
       - General settings: Ramey model, pumping losses, timestep, impedance
       - Rock properties: Thermal conductivity, density, heat capacity
       - Reservoir simulator settings: Fast mode, accuracy level, dynamic fluid properties
       - Reservoir-specific settings: ULoop, Coaxial, or DiffusionConvection specific parameters
   - **Downstream**: Power plant type, capacity, capacity factor
   - **Market**: Energy prices, market data files, resampling options
   - **Storage**: Battery and thermal energy storage (TES) parameters

3. **Validation:**
   - The app validates your configuration in real-time
   - Errors are displayed if any parameters are invalid

4. **Export Configuration:**
   - Download your configuration as a JSON file for future use

### Page 2: Simulation

The Simulation page allows you to run your configured simulation:

1. **Initialize Simulation:**
   - Click "Initialize Simulation" to create the World instance
   - Review the configuration summary

2. **Run Simulation:**
   - Click "Run Simulation" to start the simulation
   - Monitor progress with the progress bar
   - View real-time status updates

3. **Simulation Status:**
   - Idle: Ready to run
   - Running: Simulation in progress
   - Completed: Simulation finished successfully
   - Error: Simulation failed (check error message)

### Page 3: Results

The Results page displays simulation outputs:

1. **Key Metrics:**
   - NPV (Net Present Value)
   - LCOE (Levelized Cost of Energy)
   - ROI (Return on Investment)
   - PBP (Payback Period)

2. **Visualizations:**
   - Economic plots (CAPEX/OPEX breakdown)
   - Operations plots (temperature, power output, flow rates)
   - Reservoir plots (doublet visualization)

3. **Data Export:**
   - Export results to CSV or JSON
   - Download configuration files

## Example Configurations

Example configuration files are available in `examples/configs/`:

- **exampleA.json**: Simple percentage decline reservoir model
- **exampleB.json**: Binary power plant with weather data
- **exampleF.json**: Diffusion-convection reservoir model
- **exampleG.json**: Battery storage integration
- **exampleI.json**: U-Loop reservoir model
- **exampleL_Scalable_EGS.json**: Scalable EGS configuration

Example data files are available in `examples/data/sample_project/` for use with configurations that require external data files.

## Configuration File Format

Configuration files use JSON format with the following structure:

```json
{
    "L": 30,
    "d": 0.07,
    "itc": 0.0,
    "inflation": 0.02,
    "powerplant_type": "Binary",
    "powerplant_capacity": 50.0,
    "reservoir_type": "percentage",
    "well_tvd": 3000.0,
    "num_prd": 4,
    "inj_prd_ratio": 1.0,
    "ramey": true,
    "pumping": true,
    "timestep_hours": 1.0,
    "krock": 3.0,
    "rhorock": 2700.0,
    "cprock": 1000.0,
    "reservoir_simulator_settings": {
        "fast_mode": false,
        "period": 31536000,
        "accuracy": 1,
        "DynamicFluidProperties": true
    },
    ...
}
```

### Simulator Settings in JSON

You can include simulator settings directly in your JSON configuration:

```json
{
    "ramey": true,
    "pumping": true,
    "timestep_hours": 1.0,
    "krock": 3.0,
    "rhorock": 2700.0,
    "cprock": 1000.0,
    "krock_wellbore": 3.0,
    "impedance": 0.1,
    "N_ramey_mv_avg": 168,
    "reservoir_simulator_settings": {
        "fast_mode": false,
        "period": 31536000,
        "accuracy": 1,
        "DynamicFluidProperties": true
    },
    "k_m": 2.83,
    "rho_m": 2875.0,
    "c_m": 825.0,
    "k_f": 0.68,
    "mu_f": 6e-4,
    "cp_f": 4200.0,
    "rho_f": 1000.0,
    "fullyimplicit": 1,
    "FMM": 1,
    "FMMtriggertime": 864000.0
}
```

See example configuration files for complete parameter lists.

## Advanced Simulator Settings

All simulator settings are now configurable via both UI and JSON. These settings control the numerical simulation behavior:

### General Settings
- **ramey**: Enable/disable Ramey's model for wellbore heat loss/gain (default: True)
- **pumping**: Account for parasitic losses due to pumping (default: True)
- **timestep_hours**: Simulation timestep size in hours (default: 1.0)
- **impedance**: Reservoir pressure losses (default: 0.1)
- **N_ramey_mv_avg**: Timesteps for averaging Ramey's f-function (default: 168)

### Rock Properties
- **krock**: Rock thermal conductivity in W/m-K (default: 3.0)
- **rhorock**: Rock bulk density in kg/m³ (default: 2700.0)
- **cprock**: Rock heat capacity in J/kg-K (default: 1000.0)
- **krock_wellbore**: Thermal conductivity around wellbore in W/m-K (default: 3.0)

### Reservoir Simulator Settings
- **fast_mode**: Reduce computational requirements (default: False)
- **period**: Time period before reservoir state update in days (when fast_mode=True)
- **accuracy**: Numerical accuracy level 1-6 (1=fastest, 6=most accurate, default: 1)
- **DynamicFluidProperties**: Update geofluid properties using steam tables (default: True)

### ULoop-Specific Settings
- **k_m**: Rock thermal conductivity for ULoop (default: 2.83 W/m-K)
- **rho_m**: Rock density for ULoop (default: 2875.0 kg/m³)
- **c_m**: Rock heat capacity for ULoop (default: 825.0 J/kg-K)
- **k_f**: Fluid thermal conductivity (default: 0.68 W/m-K)
- **mu_f**: Fluid kinematic viscosity (default: 6e-4 m²/s)
- **cp_f**: Fluid heat capacity (default: 4200.0 J/kg-K)
- **rho_f**: Fluid density (default: 1000.0 kg/m³)
- **fullyimplicit**: Euler's method solver option (default: 1)
- **FMM**: Fast multi-pole method enable (default: 1)
- **FMMtriggertime**: FMM trigger time in days (default: 10 days)

### Coaxial-Specific Settings
Similar to ULoop but with different defaults:
- **k_m**: 2.3 W/m-K
- **k_f**: 0.6 W/m-K
- **cp_f**: 4184.0 J/kg-K
- **rho_f**: 990.0 kg/m³

### DiffusionConvection-Specific Overrides
- **krock_diffusion**: Override for diffusion_convection (default: 30.0 W/m-K)
- **rhorock_diffusion**: Override for diffusion_convection (default: 2600.0 kg/m³)
- **cprock_diffusion**: Override for diffusion_convection (default: 1100.0 J/kg-K)

All simulator settings can be configured via:
1. **UI**: Expand "Advanced Simulator Settings" in the Upstream tab
2. **JSON**: Include settings in your configuration file

## Features

- ✅ **No hardcoded values**: All parameters are configurable via UI or JSON import
- ✅ **Comprehensive forms**: Organized by category for easy navigation
- ✅ **Advanced simulator settings**: All numerical simulation parameters are user-configurable
- ✅ **Real-time validation**: Configuration validation before simulation
- ✅ **Interactive visualizations**: Plotly charts for better exploration
- ✅ **Data export**: Export results to CSV/JSON
- ✅ **JSON import/export**: Save and load configurations
- ✅ **Example configs**: Pre-configured examples for quick start
- ✅ **Progress tracking**: Real-time simulation progress updates
- ✅ **Automatic config change detection**: Simulation resets when parameters change

## Troubleshooting

### Import Errors
If you encounter import errors, ensure:
- All dependencies are installed: `pip install -r requirements.txt`
- You're running from the project root directory
- Python path includes the project directory

### Configuration Errors
- Check that all required parameters are set
- Verify file paths are correct (relative to project root or absolute)
- Ensure data files exist if referenced in configuration

### Simulation Errors
- Check the error message in the Simulation page
- Verify configuration parameters are within valid ranges
- Ensure sufficient system resources for large simulations

## Development

The application is structured for easy extension:

- **Add new parameters**: Update `streamlit_app/core/defaults.py` and `streamlit_app/ui/components/config_forms.py`
- **Add new visualizations**: Create functions in `streamlit_app/visualization/`
- **Add new pages**: Create files in `streamlit_app/pages/` (numbered for ordering)

## Support

For issues, questions, or contributions, please refer to the main README.md or contact the project maintainers.
