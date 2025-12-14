<!--
Hey, thanks for using the awesome-readme-template template.  
If you have any enhancements, then fork this project and create a pull request 
or just open an issue with the label "enhancement".

Don't forget to give this project a star for additional support ;)
Maybe you can mention me or this repo in the acknowledgements too
-->

<div align="center"> 
  <img src="docs/source/_static/fgem_logo.png" alt="logo" width="400" height="auto" />
</div>


**FGEM** (/if'gem/), Flexible Geothermal Economics Model, is an open-source Python library for evaluating lifecycle techno-economics of baseload and flexible geothermal energy projects. 
It performs sequential simulations spanning hourly to yearly timestepping using analytical, numerical, and iterative models. 
It also simulates hybrid systems involving storage facilities (e.g., thermal energy storage tanks and Lithium-ion battery units). 
For more technical details, you may refer to our [Applied Energy Journal Article](https://doi.org/10.1016/j.apenergy.2023.122125).

## 🚀 Streamlit Web Application

FGEM provides a user-friendly Streamlit web application with a wizard-style interface for configuring and running geothermal energy project simulations:

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app:**
   ```bash
   python run_streamlit.py
   ```
   Or directly:
   ```bash
   streamlit run streamlit_app/app.py
   ```

3. **Navigate through the wizard:**
   - **Configuration**: Set up your project parameters (economics, reservoir, power plant, market, storage)
   - **Simulation**: Run the simulation with progress tracking
   - **Results**: View key metrics, interactive visualizations, and export data

### Features

- ✅ **No hardcoded values**: All parameters are configurable via UI or JSON import
- ✅ **Comprehensive forms**: Organized by category (Economics, Upstream, Downstream, Market, Storage)
- ✅ **Real-time validation**: Configuration validation before simulation
- ✅ **Interactive visualizations**: Plotly charts for better exploration
- ✅ **Data export**: Export results to CSV/JSON
- ✅ **JSON import/export**: Save and load configurations

### Project Structure

```
FGEM/
├── streamlit_app/           # Main Streamlit application
│   ├── app.py              # Entry point
│   ├── core/                # Business logic (config, simulation engine, defaults)
│   ├── ui/                  # UI components (forms, file uploads, progress)
│   ├── pages/               # Wizard pages (Configuration, Simulation, Results)
│   ├── visualization/       # Plotting functions (separated from calculations)
│   └── fgem/                # Core simulation module (world, subsurface, powerplant, etc.)
├── examples/
│   ├── configs/             # Example configuration files (JSON)
│   └── data/                # Example data files (CSV)
├── requirements.txt         # Python dependencies
├── run_streamlit.py         # Helper script to launch the app
└── README.md               # This file
```

### Architecture

The Streamlit app follows a clean separation of concerns:
- **streamlit_app/core/**: Business logic (config management, simulation engine, defaults)
- **streamlit_app/ui/**: User interface components (forms, file uploads, progress indicators)
- **streamlit_app/visualization/**: Plotting functions (separated from calculations)
- **streamlit_app/pages/**: Wizard pages (Configuration, Simulation, Results)
- **streamlit_app/fgem/**: Core simulation module (world, subsurface, powerplant, markets, weather, storage)

<br />
<div align="center">
  <img src="docs/source/_static/flowchart.png" alt="flowchart" width="500" height="auto" />
</div>
 <br />

# Contributing

We welcome your contributions to this project. Please see the [contributions](https://fgem.readthedocs.io/en/latest/reference_contribution.html) guide in our readthedocs page for more information. Please do not hesitate to contact aljubrmj@stanford.edu with specific questions, requests, or feature ideas.

# License

The project is licensed under MIT License. See LICENSE.txt for more information.

# Citation

Please cite our Journal Article:

```
@article{aljubran2024fgem,
  title={FGEM: Flexible Geothermal Economics Modeling tool},
  author={Aljubran, MJ and Horne, Roland N},
  journal={Applied Energy},
  volume={353},
  pages={122125},
  year={2024},
  publisher={Elsevier}
}
```
