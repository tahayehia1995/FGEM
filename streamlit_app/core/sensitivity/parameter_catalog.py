"""Parameter catalog for sensitivity analysis.

This module provides a comprehensive, user-friendly registry for FGEM configuration
parameters with metadata such as units, categories, suggested ranges, and choices.

Catalog entries may refer to:
- top-level scalar keys (e.g., `well_tvd`)
- list-like parameters exposed as indexed pseudo-keys (e.g., `battery_duration_0`)
- nested dict settings exposed as dotted pseudo-keys (e.g., `reservoir_simulator_settings.fast_mode`)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..config_manager import ConfigManager


def _infer_kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    return "categorical"


def build_parameter_catalog(
    *,
    defaults: Dict[str, Any],
    config_manager: Optional[ConfigManager] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build a parameter catalog keyed by parameter name / pseudo-key."""

    cm = config_manager or ConfigManager()

    # Categories for major groups (used for filtering in the UI)
    econ = {
        "L",
        "d",
        "itc",
        "inflation",
        "contingency",
        "drilling_cost",
        "powerplant_interconnection_cost",
        "exploration_cost_intercept",
        "exploration_cost_slope",
        "stimulation_cost",
    }
    upstream = {
        "reservoir_type",
        "reservoir_filename",
        "Tres_init",
        "Pres_init",
        "V_res",
        "phi_res",
        "res_thickness",
        "krock",
        "cprock",
        "drawdp",
        "plateau_length",
        "rock_energy_recovery",
        "geothermal_gradient",
        "redrill_ratio",
        "shutoff_ratio",
        "total_drilling_length",
        "prd_total_drilling_length",
        "inj_total_drilling_length",
        # wells
        "well_tvd",
        "well_md",
        "prd_well_diam",
        "inj_well_diam",
        "int_well_diam",
        "lateral_length",
        "lateral_diam",
        "lateral_spacing",
        "numberoflaterals",
        "num_prd",
        "inj_prd_ratio",
        "waterloss",
        "pumpeff",
        "DSR",
        "SSR",
        "PI",
        "II",
        "PumpingModel",
        "closedloop_design",
        "dx",
        # coaxial
        "casing_inner_diam",
        "tube_inner_diam",
        "tube_thickness",
        "k_tube",
        "coaxialflowtype",
    }
    downstream = {
        "powerplant_capacity",
        "powerplant_type",
        "pipinglength",
        "cf",
        "bypass",
        "powerplant_opex_rate",
        "powerplant_labor_rate",
        "powerplant_usd_per_kw_min",
        "Tres_pp_design",
        "m_prd_pp_design",
        "powerplant_k",
    }
    market = {
        "energy_price",
        "energy_market_filename",
        "capacity_price",
        "capacity_market_filename",
        "recs_price",
        "ppa_price",
        "ppa_escalaction_rate",
        "fat_factor",
        "resample",
        "oversample_first_day",
    }
    weather = {
        "weather_filename",
        "surface_temp",
        "sup3rcc_weather_forecast",
    }
    storage = {
        "battery_costs_filename",
        "battery_interconnection_cost",
        "battery_energy_cost",
        "battery_power_cost",
        "battery_fom",
        "battery_energy_augmentation",
        "battery_power_augmentation",
        "battery_elcc",
        "battery_roundtrip_eff",
        "battery_lifetime",
        "tank_diameter",
        "tank_height",
        "tank_cost",
        # list-like
        "battery_duration",
        "battery_power_capacity",
    }
    simulator = {
        "ramey",
        "pumping",
        "timestep_hours",
        "krock_wellbore",
        "impedance",
        "N_ramey_mv_avg",
        "k_m",
        "rho_m",
        "c_m",
        "k_f",
        "mu_f",
        "cp_f",
        "rho_f",
        "fullyimplicit",
        "FMM",
        "FMMtriggertime",
        "krock_diffusion",
        "rhorock_diffusion",
        "cprock_diffusion",
        # nested dict (exposed separately)
        "reservoir_simulator_settings",
    }

    labels = {
        # economics
        "L": "Project lifetime",
        "d": "Discount rate",
        "itc": "Investment tax credit",
        "inflation": "Inflation rate",
        "contingency": "Contingency factor",
        "drilling_cost": "Drilling cost",
        # upstream
        "Tres_init": "Initial reservoir temperature",
        "Pres_init": "Initial reservoir pressure",
        "V_res": "Reservoir volume",
        "phi_res": "Reservoir porosity",
        "res_thickness": "Reservoir thickness",
        "krock": "Rock thermal conductivity",
        "cprock": "Rock heat capacity",
        "geothermal_gradient": "Geothermal gradient",
        "well_tvd": "Well true vertical depth",
        "lateral_length": "Lateral length",
        "numberoflaterals": "Number of laterals",
        "num_prd": "Number of production wells",
        "inj_prd_ratio": "Injection/production ratio",
        "m_prd": "Production mass flow rate",
        # downstream
        "powerplant_type": "Power plant type",
        "powerplant_capacity": "Power plant capacity",
        "cf": "Capacity factor",
        "pipinglength": "Piping length",
        "bypass": "Enable bypass",
        # market
        "energy_price": "Energy price",
        "capacity_price": "Capacity price",
        "recs_price": "RECs price",
        "ppa_price": "PPA price",
        "fat_factor": "Fat factor",
        "resample": "Market resample frequency",
        # storage
        "battery_roundtrip_eff": "Battery roundtrip efficiency",
        "battery_lifetime": "Battery lifetime",
        "tank_diameter": "TES tank diameter",
        "tank_height": "TES tank height",
    }

    units = {
        "L": "years",
        "d": "fraction",
        "itc": "fraction",
        "inflation": "fraction",
        "contingency": "fraction",
        "drilling_cost": "MM$/well (<=200) or USD/m (>200)",
        "Tres_init": "°C",
        "Pres_init": "bar",
        "V_res": "km³",
        "phi_res": "fraction",
        "res_thickness": "m",
        "krock": "W/m-K",
        "cprock": "J/kg-K",
        "geothermal_gradient": "°C/km",
        "well_tvd": "m",
        "lateral_length": "m",
        "lateral_spacing": "m",
        "prd_well_diam": "m",
        "inj_well_diam": "m",
        "m_prd": "kg/s",
        "powerplant_capacity": "MWe",
        "pipinglength": "km",
        "energy_price": "$/MWh",
        "capacity_price": "$/MW-hour",
        "recs_price": "$/MWh",
        "ppa_price": "$/MWh",
        "tank_diameter": "m",
        "tank_height": "m",
        "battery_power_capacity_0": "MWe",
        "battery_power_capacity_1": "MWe",
        "battery_duration_0": "hours",
        "battery_duration_1": "hours",
    }

    suggested_ranges = {
        "d": (0.0, 0.2),
        "itc": (0.0, 0.5),
        "inflation": (0.0, 0.1),
        "contingency": (0.0, 0.5),
        "Tres_init": (50.0, 350.0),
        "Pres_init": (1.0, 250.0),
        "V_res": (0.1, 50.0),
        "phi_res": (0.01, 0.4),
        "well_tvd": (500.0, 9000.0),
        "lateral_length": (0.0, 10000.0),
        "m_prd": (10.0, 400.0),
        "powerplant_capacity": (1.0, 200.0),
        "energy_price": (0.0, 200.0),
        "capacity_price": (0.0, 500.0),
        "recs_price": (0.0, 200.0),
    }

    # Categorical choices (from validation logic + UI)
    reservoir_types = [
        "diffusion_convection",
        "energy_decline",
        "uloop",
        "coaxial",
        "percentage",
        "tabular",
    ]
    powerplant_types = ["Binary", "Flash", "ORC", "GEOPHIRES", "HighEnthalpyCLGWGPowerPlant"]
    pumping_models = ["OpenLoop", "ClosedLoop"]
    resample_choices = ["None", "1Y", "1M", "1W", "1D", "4H", "2H", "1H", "30min", "15min"]

    catalog: Dict[str, Dict[str, Any]] = {}

    def _category_for_key(key: str) -> str:
        if key in econ:
            return "Economics"
        if key in upstream:
            return "Upstream"
        if key in downstream:
            return "Downstream"
        if key in market:
            return "Market"
        if key in weather:
            return "Weather"
        if key in storage:
            return "Storage"
        if key in simulator:
            return "SimulatorSettings"
        return "Other"

    # Core scalar keys
    for k, v in defaults.items():
        if isinstance(v, (list, tuple, dict)):
            continue
        kind = _infer_kind(v)
        if k in ("reservoir_type", "powerplant_type", "PumpingModel", "resample"):
            kind = "categorical"

        entry: Dict[str, Any] = {
            "key": k,
            "path": k,  # top-level scalar
            "category": _category_for_key(k),
            "kind": kind,
            "label": labels.get(k, k),
            "units": units.get(k, ""),
            "default": v,
        }
        if k in suggested_ranges:
            entry["suggested_min"], entry["suggested_max"] = suggested_ranges[k]
        catalog[k] = entry

    # List-like storage parameters exposed as indexed pseudo-keys
    catalog["battery_duration_0"] = {
        "key": "battery_duration_0",
        "path": ("battery_duration", 0),
        "category": "Storage",
        "kind": "float",
        "label": "Battery duration (unit 1)",
        "units": units.get("battery_duration_0", "hours"),
        "default": (defaults.get("battery_duration") or [0, 0])[0],
        "suggested_min": 0.0,
        "suggested_max": 12.0,
    }
    catalog["battery_duration_1"] = {
        "key": "battery_duration_1",
        "path": ("battery_duration", 1),
        "category": "Storage",
        "kind": "float",
        "label": "Battery duration (unit 2)",
        "units": units.get("battery_duration_1", "hours"),
        "default": (defaults.get("battery_duration") or [0, 0])[1],
        "suggested_min": 0.0,
        "suggested_max": 12.0,
    }
    catalog["battery_power_capacity_0"] = {
        "key": "battery_power_capacity_0",
        "path": ("battery_power_capacity", 0),
        "category": "Storage",
        "kind": "float",
        "label": "Battery power capacity (unit 1)",
        "units": units.get("battery_power_capacity_0", "MWe"),
        "default": (defaults.get("battery_power_capacity") or [0, 0])[0],
        "suggested_min": 0.0,
        "suggested_max": 200.0,
    }
    catalog["battery_power_capacity_1"] = {
        "key": "battery_power_capacity_1",
        "path": ("battery_power_capacity", 1),
        "category": "Storage",
        "kind": "float",
        "label": "Battery power capacity (unit 2)",
        "units": units.get("battery_power_capacity_1", "MWe"),
        "default": (defaults.get("battery_power_capacity") or [0, 0])[1],
        "suggested_min": 0.0,
        "suggested_max": 200.0,
    }

    # Nested reservoir_simulator_settings exposed as dotted pseudo-keys
    rss_default = defaults.get("reservoir_simulator_settings") or {}
    if isinstance(rss_default, dict):
        for subkey, subval in rss_default.items():
            pseudo = f"reservoir_simulator_settings.{subkey}"
            catalog[pseudo] = {
                "key": pseudo,
                "path": ("reservoir_simulator_settings", subkey),
                "category": "SimulatorSettings",
                "kind": _infer_kind(subval),
                "label": f"Reservoir simulator setting: {subkey}",
                "units": "",
                "default": subval,
            }

    # Enrich categorical choices
    if "reservoir_type" in catalog:
        catalog["reservoir_type"]["choices"] = reservoir_types
    if "powerplant_type" in catalog:
        catalog["powerplant_type"]["choices"] = powerplant_types
    if "PumpingModel" in catalog:
        catalog["PumpingModel"]["choices"] = pumping_models
    if "resample" in catalog:
        catalog["resample"]["choices"] = resample_choices

    # Some booleans should be treated as bool even if defaults came in differently
    for b in ("bypass", "ramey", "pumping", "sup3rcc_weather_forecast"):
        if b in catalog:
            catalog[b]["kind"] = "bool"

    return catalog

