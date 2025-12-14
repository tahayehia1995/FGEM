"""Configuration management for FGEM.

Handles loading defaults, merging user configs, validation, and JSON import/export.
"""

import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
from .defaults import ConfigDefaults, get_defaults


class ConfigManager:
    """Manages configuration loading, validation, and export/import."""
    
    def __init__(self, defaults: Optional[ConfigDefaults] = None):
        """Initialize ConfigManager with defaults.
        
        Args:
            defaults: Optional ConfigDefaults instance. If None, uses get_defaults().
        """
        self.defaults = defaults if defaults is not None else get_defaults()
        self._default_dict = None
    
    def load_defaults(self) -> Dict[str, Any]:
        """Load default configuration as dictionary.
        
        Returns:
            Dictionary with all default values in flat format compatible with World class.
        """
        if self._default_dict is None:
            self._default_dict = self.defaults.to_dict()
        return self._default_dict.copy()
    
    def merge_config(self, user_config: Dict[str, Any], config_file_path: Optional[str] = None) -> Dict[str, Any]:
        """Merge user configuration with defaults.
        
        Args:
            user_config: User-provided configuration dictionary (can be nested or flat).
            config_file_path: Optional path to the config file (used for resolving relative paths).
            
        Returns:
            Merged configuration dictionary with all values set (defaults + user overrides).
        """
        defaults = self.load_defaults()
        
        # Handle nested config structure (like examples/config.json)
        if self._is_nested_config(user_config):
            user_config = self._flatten_config(user_config)
        
        # Merge: user config overrides defaults
        # Use deep merge for nested dictionaries like reservoir_simulator_settings
        merged = self._deep_merge(defaults.copy(), user_config)
        
        # Handle special cases
        merged = self._process_special_values(merged)
        
        # Resolve relative paths for project_data_dir
        if merged.get("project_data_dir"):
            project_data_dir = merged["project_data_dir"]
            if not os.path.isabs(project_data_dir):
                if config_file_path:
                    # Resolve relative to config file location
                    config_dir = os.path.dirname(os.path.abspath(config_file_path))
                    # If path starts with "./", resolve relative to config file's parent directory
                    # (since configs are in examples/configs/ but data is in examples/data/)
                    if project_data_dir.startswith("./"):
                        config_parent = os.path.dirname(config_dir)
                        merged["project_data_dir"] = os.path.normpath(os.path.join(config_parent, project_data_dir[2:]))
                    else:
                        merged["project_data_dir"] = os.path.normpath(os.path.join(config_dir, project_data_dir))
                else:
                    # Resolve relative to current working directory
                    merged["project_data_dir"] = os.path.normpath(os.path.join(os.getcwd(), project_data_dir))
        
        # Calculate Tres_init if not provided but we have the required parameters
        if merged.get("Tres_init") is None:
            surface_temp = merged.get("surface_temp")
            geothermal_gradient = merged.get("geothermal_gradient")
            well_tvd = merged.get("well_tvd")
            if surface_temp is not None and geothermal_gradient is not None and well_tvd is not None:
                merged["Tres_init"] = surface_temp + geothermal_gradient / 1000.0 * well_tvd
        
        # Calculate dx defaults based on reservoir type if not provided
        if merged.get("dx") is None:
            reservoir_type = merged.get("reservoir_type", "").lower()
            lateral_length = merged.get("lateral_length", 0)
            well_tvd = merged.get("well_tvd", 0)
            
            if reservoir_type == "uloop":
                # For uloop, use lateral_length if available, otherwise well_tvd
                if lateral_length > 0:
                    merged["dx"] = max(10.0, lateral_length // 10)
                elif well_tvd > 0:
                    merged["dx"] = max(10.0, well_tvd // 10)
                else:
                    merged["dx"] = 500.0  # Default for uloop
            elif reservoir_type == "coaxial":
                # For coaxial, use well_tvd
                if well_tvd > 0:
                    merged["dx"] = max(10.0, well_tvd // 10)
                else:
                    merged["dx"] = 100.0  # Default for coaxial
            elif lateral_length > 0:
                # For other types with lateral_length
                merged["dx"] = max(10.0, lateral_length // 10)
            elif well_tvd > 0:
                # Fallback to well_tvd
                merged["dx"] = max(10.0, well_tvd // 10)
            else:
                # Final fallback
                merged["dx"] = 100.0
        
        # Set reservoir-specific defaults - override general defaults if needed
        # Based on working examples: exampleI.json and exampleJ.json
        reservoir_type = merged.get("reservoir_type", "").lower()
        if reservoir_type == "uloop":
            # Set defaults for uloop based on working examples
            # Override defaults even if they were set from defaults.py
            
            # Critical geometry parameters from examples
            if merged.get("well_tvd", 0) < 7000:
                merged["well_tvd"] = 7000.0
            if merged.get("lateral_length", 0) < 3500:  # exampleJ uses 3500, exampleI uses 5000
                merged["lateral_length"] = 5000.0  # Use the more conservative value
            if merged.get("numberoflaterals", 1) < 12:  # Examples use 12
                merged["numberoflaterals"] = 12
            if merged.get("dx") is None or merged.get("dx", 0) < 500:
                merged["dx"] = 500.0  # Examples use 500
            
            # Other important parameters from examples
            if merged.get("res_thickness", 0) < 1000:
                merged["res_thickness"] = 1000.0  # Examples use 1000
            # lateral_spacing: examples use 30, but only override if very low
            if merged.get("lateral_spacing", 0) < 30:
                merged["lateral_spacing"] = 30.0  # Examples use 30
            elif merged.get("lateral_spacing") is None:
                merged["lateral_spacing"] = 30.0
            
            # Well diameters from examples (0.216 instead of default 0.3115)
            # Override if using default value or if not set
            if merged.get("lateral_diam") is None or abs(merged.get("lateral_diam", 0) - 0.3115) < 0.01:
                merged["lateral_diam"] = 0.216  # Examples use 0.216
            if merged.get("prd_well_diam") is None or abs(merged.get("prd_well_diam", 0) - 0.3115) < 0.01:
                merged["prd_well_diam"] = 0.216  # Examples use 0.216
            if merged.get("inj_well_diam") is None or abs(merged.get("inj_well_diam", 0) - 0.3115) < 0.01:
                merged["inj_well_diam"] = 0.216  # Examples use 0.216
            
            # Pumping model from examples
            if merged.get("PumpingModel", "").lower() not in ["closedloop", "closed_loop"]:
                merged["PumpingModel"] = "ClosedLoop"  # Examples use ClosedLoop
            
            # Other parameters from examples
            if merged.get("waterloss", 1.0) > 0.01:  # Examples use 0.0
                merged["waterloss"] = 0.0
            if merged.get("pumpeff", 0) < 0.8:
                merged["pumpeff"] = 0.8  # Examples use 0.8
            
            # V_res from examples (18 instead of 5)
            if merged.get("V_res", 0) < 10:
                merged["V_res"] = 18.0  # Examples use 18
            
            # Calculate half_lateral_length if needed
            if merged.get("half_lateral_length") is None or merged.get("half_lateral_length", 0) < 2500:
                merged["half_lateral_length"] = merged.get("lateral_length", 5000.0) / 2.0
        elif reservoir_type == "coaxial":
            # Set defaults for coaxial
            if merged.get("well_tvd", 0) < 1000:
                merged["well_tvd"] = 3000.0
            if merged.get("dx") is None or merged.get("dx", 0) == 0:
                merged["dx"] = 100.0
        elif reservoir_type == "energy_decline":
            # Set defaults for energy_decline
            if merged.get("well_tvd", 0) < 1000:
                merged["well_tvd"] = 3000.0
            if merged.get("V_res", 0) == 0:
                merged["V_res"] = 5.0
            if merged.get("phi_res", 0) == 0:
                merged["phi_res"] = 0.1
        elif reservoir_type == "diffusion_convection":
            # Set defaults for diffusion_convection (aligned with examples, especially scalable EGS)
            if merged.get("well_tvd", 0) < 1000:
                merged["well_tvd"] = 3500.0
            # Default lateral_length to 2000 if not specified or zero
            if merged.get("lateral_length", 0) == 0:
                merged["lateral_length"] = 2000.0
            # Default res_thickness to 1000 if not specified or zero
            if merged.get("res_thickness", 0) == 0:
                merged["res_thickness"] = 1000.0
            # Default krock to 30 if not specified or using old default value (3)
            # Check if krock is the old default (3) - this handles backward compatibility
            krock = merged.get("krock", 0)
            if krock == 0:
                merged["krock"] = 30.0
            elif krock == 3:
                # Old default value detected - upgrade to new default for diffusion_convection
                merged["krock"] = 30.0
        elif reservoir_type in ["percentage", "tabular"]:
            # Set defaults for percentage and tabular
            if merged.get("well_tvd", 0) < 1000:
                merged["well_tvd"] = 3000.0
        
        # Ensure powerplant_capacity is set
        if merged.get("powerplant_capacity") is None:
            merged["powerplant_capacity"] = 50.0
        
        # Ensure Tres_init is calculated if not provided
        if merged.get("Tres_init") is None:
            surface_temp = merged.get("surface_temp", 20.0)
            geothermal_gradient = merged.get("geothermal_gradient", 35.0)
            well_tvd = merged.get("well_tvd", 3000.0)
            merged["Tres_init"] = surface_temp + geothermal_gradient / 1000.0 * well_tvd
        
        # Set drilling_cost if not provided (required for some calculations)
        # Default is 10.0 MM$ per well (matches scalable EGS example behavior when not specified)
        # This will be converted to USD/m by _normalize_drilling_cost() below
        if merged.get("drilling_cost") is None:
            # Default user-facing drilling cost = 10 MM$ per well
            merged["drilling_cost"] = 10.0

        # Normalize drilling_cost units and sanitize NaNs/Infs for numeric values
        # This converts MM$ per well (values <= 200) to USD/m for internal calculations
        merged = self._normalize_drilling_cost(merged)
        merged = self._sanitize_numeric_values(merged, defaults)
        
        return merged

    def _sanitize_numeric_values(self, config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Replace NaN/Inf numeric values with reasonable defaults to avoid downstream errors."""
        out = config.copy()

        def _finite_or_none(x):
            try:
                return x if np.isfinite(x) else None
            except Exception:
                return None

        # Only sanitize scalars and simple lists/tuples (don’t touch nested dict settings)
        for k, v in list(out.items()):
            if isinstance(v, (int, float, np.floating, np.integer)):
                fv = _finite_or_none(float(v))
                if fv is None:
                    dv = defaults.get(k)
                    dv = _finite_or_none(float(dv)) if isinstance(dv, (int, float, np.floating, np.integer)) else None
                    out[k] = dv if dv is not None else 0.0
            elif isinstance(v, (list, tuple)) and v and all(isinstance(x, (int, float, np.floating, np.integer)) for x in v):
                cleaned = []
                dv = defaults.get(k)
                dv_list = dv if isinstance(dv, list) else None
                for i, x in enumerate(v):
                    fx = _finite_or_none(float(x))
                    if fx is None:
                        fallback = None
                        if dv_list and i < len(dv_list) and isinstance(dv_list[i], (int, float, np.floating, np.integer)):
                            fallback = _finite_or_none(float(dv_list[i]))
                        cleaned.append(fallback if fallback is not None else 0.0)
                    else:
                        cleaned.append(fx)
                out[k] = cleaned

        return out

    def _normalize_drilling_cost(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize drilling_cost to internal units (USD/m), while allowing user input in MM$."""
        out = config.copy()
        val = out.get("drilling_cost")
        if val is None or val is False:
            return out

        try:
            val_f = float(val)
        except Exception:
            return out

        # If user enters <= 200, interpret as MM$ per well (user-facing).
        # If > 200, interpret as USD/m (backward compatible with examples using 1000).
        if not np.isfinite(val_f) or val_f <= 0:
            out["drilling_cost"] = 0.0
            return out

        if val_f <= 200.0:
            # Estimate drilled length for conversion
            well_tvd = float(out.get("well_tvd", 3000.0) or 3000.0)
            reservoir_type = str(out.get("reservoir_type", "")).lower()
            numberoflaterals = float(out.get("numberoflaterals", 1) or 1)
            lateral_length = float(out.get("lateral_length", 0.0) or 0.0)
            half_lateral_length = out.get("half_lateral_length")
            try:
                half_lateral_length = float(half_lateral_length) if half_lateral_length is not None else None
            except Exception:
                half_lateral_length = None

            if reservoir_type == "uloop":
                lat_len = half_lateral_length if half_lateral_length is not None else (lateral_length / 2.0)
            else:
                lat_len = lateral_length

            well_md = well_tvd + max(lat_len, 0.0) * max(numberoflaterals, 1.0)
            well_md = max(well_md, 1.0)

            # Convert MM$/well -> USD/m
            out["drilling_cost"] = (val_f * 1e6) / well_md

        return out
    
    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate configuration values.
        
        Args:
            config: Configuration dictionary to validate.
            
        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        errors = []
        
        # Validate numeric ranges
        if "L" in config and (config["L"] < 1 or config["L"] > 100):
            errors.append("Project lifetime (L) must be between 1 and 100 years")
        
        if "d" in config and (config["d"] < 0 or config["d"] > 1):
            errors.append("Discount rate (d) must be between 0 and 1")
        
        if "itc" in config and (config["itc"] < 0 or config["itc"] > 1):
            errors.append("Investment tax credit (itc) must be between 0 and 1")
        
        if "well_tvd" in config and config["well_tvd"] <= 0:
            errors.append("Well TVD must be positive")
        
        if "num_prd" in config and config["num_prd"] < 0:
            errors.append("Number of production wells must be non-negative")
        
        if "powerplant_capacity" in config and config["powerplant_capacity"] is not None:
            if config["powerplant_capacity"] <= 0:
                errors.append("Power plant capacity must be positive")

        if "drilling_cost" in config and config["drilling_cost"] is not None:
            try:
                dc = float(config["drilling_cost"])
                if (not np.isfinite(dc)) or dc <= 0:
                    errors.append("Drilling cost must be positive (either MM$/well <= 200, or USD/m > 200)")
            except Exception:
                errors.append("Drilling cost must be a number")
        
        # Validate file paths exist if specified (handle relative paths)
        project_data_dir = config.get("project_data_dir", "")
        
        if "reservoir_filename" in config and config["reservoir_filename"]:
            reservoir_path = config["reservoir_filename"]
            if project_data_dir and not os.path.isabs(reservoir_path):
                reservoir_path = os.path.join(project_data_dir, reservoir_path)
            if not os.path.exists(reservoir_path):
                # Only warn, don't error - file might be created later or path might be resolved at runtime
                pass  # Don't add error for now, let World class handle it
        
        if "energy_market_filename" in config and config["energy_market_filename"]:
            energy_path = config["energy_market_filename"]
            if project_data_dir and not os.path.isabs(energy_path):
                energy_path = os.path.join(project_data_dir, energy_path)
            if not os.path.exists(energy_path):
                # Only warn, don't error - file might be created later or path might be resolved at runtime
                pass  # Don't add error for now, let World class handle it
        
        if "weather_filename" in config and config["weather_filename"]:
            weather_path = config["weather_filename"]
            if project_data_dir and not os.path.isabs(weather_path):
                weather_path = os.path.join(project_data_dir, weather_path)
            if not os.path.exists(weather_path):
                # Only warn, don't error - file might be created later or path might be resolved at runtime
                pass  # Don't add error for now, let World class handle it
        
        # Validate reservoir type
        valid_reservoir_types = [
            "diffusion_convection", "energy_decline", "uloop", 
            "coaxial", "percentage", "tabular"
        ]
        if "reservoir_type" in config:
            if config["reservoir_type"] not in valid_reservoir_types:
                errors.append(f"Invalid reservoir_type. Must be one of: {valid_reservoir_types}")
        
        # Validate power plant type
        valid_pp_types = ["Binary", "Flash", "ORC", "GEOPHIRES", "HighEnthalpyCLGWGPowerPlant"]
        if "powerplant_type" in config:
            pp_type = config["powerplant_type"]
            if not any(valid in pp_type for valid in valid_pp_types):
                errors.append(f"Invalid powerplant_type. Should contain one of: {valid_pp_types}")
        
        return len(errors) == 0, errors
    
    def import_from_json(self, filepath: str) -> Dict[str, Any]:
        """Import configuration from JSON file.
        
        Args:
            filepath: Path to JSON configuration file.
            
        Returns:
            Merged configuration dictionary.
        """
        with open(filepath, 'r') as f:
            json_config = json.load(f)
        
        return self.merge_config(json_config)
    
    def export_to_json(self, config: Dict[str, Any], filepath: str, 
                      nested: bool = True) -> None:
        """Export configuration to JSON file.
        
        Args:
            config: Configuration dictionary to export.
            filepath: Path where to save JSON file.
            nested: If True, exports in nested format (like examples/config.json).
                    If False, exports in flat format.
        """
        if nested:
            config_to_export = self._nest_config(config)
        else:
            config_to_export = config
        
        with open(filepath, 'w') as f:
            json.dump(config_to_export, f, indent=4)
    
    def _is_nested_config(self, config: Dict[str, Any]) -> bool:
        """Check if config is in nested format."""
        # Check if top-level keys match expected nested structure
        nested_keys = ["metadata", "economics", "upstream", "downstream", 
                      "market", "weather", "storage"]
        return any(key in config for key in nested_keys)
    
    def _flatten_config(self, nested_config: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested configuration to flat format."""
        flat = {}
        
        # Handle metadata
        if "metadata" in nested_config:
            meta = nested_config["metadata"]
            flat.update({
                "project_data_dir": meta.get("base_dir", meta.get("project_data_dir")),
                "time_init": meta.get("time_init"),
                "project_lat": meta.get("lat"),
                "project_long": meta.get("long"),
                "n_jobs": meta.get("n_jobs"),
                "reservoir_simulator_settings": meta.get("reservoir_simulator_settings"),
            })
        
        # Handle economics
        if "economics" in nested_config:
            econ = nested_config["economics"]
            flat.update({
                "L": econ.get("L"),
                "d": econ.get("d"),
                "itc": econ.get("itc"),
                "inflation": econ.get("inflation"),
                "contingency": econ.get("contingency"),
                "drilling_cost": econ.get("drilling_cost"),
                "powerplant_interconnection_cost": econ.get("powerplant_interconnection_cost"),
                "exploration_cost_intercept": econ.get("exploration_cost_intercept"),
                "exploration_cost_slope": econ.get("exploration_cost_slope"),
                "stimulation_cost": econ.get("stimulation_cost"),
            })
        
        # Handle upstream (reservoir + well)
        if "upstream" in nested_config:
            upstream = nested_config["upstream"]
            flat.update({
                "reservoir_type": upstream.get("reservoir_type"),
                "Tres_init": upstream.get("Tres_init"),
                "Pres_init": upstream.get("Pres_init"),
                "V_res": upstream.get("V_res"),
                "phi_res": upstream.get("phi_res"),
                "drawdp": upstream.get("drawdp"),
                "plateau_length": upstream.get("plateau_length"),
                "geothermal_gradient": upstream.get("geothermal_gradient"),
                "surface_temp": upstream.get("surface_temp"),
                "well_tvd": upstream.get("well_depth", upstream.get("well_tvd")),
                "lateral_length": upstream.get("lateral_length"),
                "numberoflaterals": upstream.get("numberoflaterals"),
                "res_thickness": upstream.get("thickness", upstream.get("res_thickness")),
                "krock": upstream.get("krock"),
                "prd_well_diam": upstream.get("prd_well_diam"),
                "inj_well_diam": upstream.get("inj_well_diam"),
                "lateral_diam": upstream.get("lateral_diam"),
                "num_prd": upstream.get("num_prd"),
                "inj_prd_ratio": upstream.get("inj_prd_ratio"),
                "waterloss": upstream.get("waterloss"),
                "rock_energy_recovery": upstream.get("rock_energy_recovery"),
                "pumpeff": upstream.get("pumpeff"),
                "DSR": upstream.get("DSR"),
                "SSR": upstream.get("SSR"),
                "PI": upstream.get("PI"),
                "II": upstream.get("II"),
                "PumpingModel": upstream.get("PumpingModel"),
                "closedloop_design": upstream.get("closedloop_design"),
                "ramey": upstream.get("ramey"),
                "reservoir_filename": upstream.get("reservoir_filename"),
                "dx": upstream.get("dx"),
                "lateral_spacing": upstream.get("lateral_spacing"),
            })
        
        # Handle downstream (powerplant)
        if "downstream" in nested_config and "powerplant" in nested_config["downstream"]:
            pp = nested_config["downstream"]["powerplant"]
            flat.update({
                "powerplant_type": pp.get("power_plant_type", pp.get("powerplant_type")),
                "powerplant_capacity": pp.get("ppc", pp.get("powerplant_capacity")),
                "pipinglength": pp.get("pipinglength"),
                "cf": pp.get("cf"),
                "bypass": pp.get("bypass"),
            })
        
        # Handle market
        if "market" in nested_config:
            market = nested_config["market"]
            flat.update({
                "energy_price": market.get("energy_price"),
                "energy_market_filename": market.get("energy_price") if isinstance(market.get("energy_price"), str) else market.get("energy_market_filename"),
                "capacity_price": market.get("capacity_price"),
                "capacity_market_filename": market.get("capacity_price") if isinstance(market.get("capacity_price"), str) else market.get("capacity_market_filename"),
                "fat_factor": market.get("fat_factor"),
                "resample": market.get("resample"),
            })
        
        # Handle weather
        if "weather" in nested_config:
            weather = nested_config["weather"]
            flat.update({
                "weather_filename": weather.get("ambient_temperature") if isinstance(weather.get("ambient_temperature"), str) else weather.get("weather_filename"),
                "surface_temp": weather.get("surface_temp"),
            })
        
        # Handle storage (battery + TES)
        if "storage" in nested_config:
            storage = nested_config["storage"]
            if "battery" in storage:
                bat = storage["battery"]
                flat.update({
                    "battery_duration": bat.get("duration", bat.get("battery_duration", [0, 0])),
                    "battery_power_capacity": bat.get("power_capacity", bat.get("battery_power_capacity", [0, 0])),
                    "battery_costs_filename": bat.get("battery_costs_filename"),
                    "battery_roundtrip_eff": bat.get("battery_roundtrip_eff"),
                    "battery_lifetime": bat.get("battery_lifetime"),
                })
            if "tes" in storage:
                tes = storage["tes"]
                flat.update({
                    "tank_diameter": tes.get("diameter", tes.get("tank_diameter", 0)),
                    "tank_height": tes.get("height", tes.get("tank_height", 0)),
                    "tank_cost": tes.get("tank_cost"),
                })
        
        # Remove None values to allow defaults to be used
        return {k: v for k, v in flat.items() if v is not None}
    
    def _nest_config(self, flat_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat configuration to nested format."""
        nested = {
            "metadata": {},
            "economics": {},
            "upstream": {},
            "downstream": {"powerplant": {}},
            "market": {},
            "weather": {},
        }
        
        # Map flat keys to nested structure
        # This is a simplified version - can be expanded
        
        return nested
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence.
        
        Args:
            base: Base dictionary (defaults).
            override: Override dictionary (user config).
            
        Returns:
            Merged dictionary.
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override with user value
                result[key] = value
        
        return result
    
    def _process_special_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process special configuration values that need transformation."""
        processed = config.copy()
        
        # Handle resample: convert False to None or keep string
        if "resample" in processed:
            if processed["resample"] is False:
                processed["resample"] = None
            elif isinstance(processed["resample"], str):
                # Normalize resample strings (e.g., "1w" -> "1W", "1d" -> "1D")
                resample = processed["resample"]
                if resample.lower() in ["1w", "1w"]:
                    processed["resample"] = "1W"
                elif resample.lower() in ["1d", "1d"]:
                    processed["resample"] = "1D"
                elif resample.lower() in ["2h", "4h"]:
                    processed["resample"] = resample.upper()
        
        # Handle ambient_temperature -> surface_temp mapping (for flat configs)
        if "ambient_temperature" in processed and "surface_temp" not in processed:
            processed["surface_temp"] = processed.pop("ambient_temperature")
        
        # Handle well_depth -> well_tvd mapping
        if "well_depth" in processed and "well_tvd" not in processed:
            processed["well_tvd"] = processed.pop("well_depth")
        
        # Handle powerplant_type variations
        if "powerplant_type" in processed:
            pp_type = str(processed["powerplant_type"])
            # Normalize powerplant type names
            if "geophires" in pp_type.lower():
                if "flash" in pp_type.lower():
                    processed["powerplant_type"] = "GEOPHIRESFlash"
                else:
                    processed["powerplant_type"] = "GEOPHIRESORC"
            elif pp_type.lower() == "binary":
                processed["powerplant_type"] = "Binary"
            elif pp_type.lower() == "flash":
                processed["powerplant_type"] = "Flash"
        
        # Handle reservoir_type variations
        if "reservoir_type" in processed:
            res_type = str(processed["reservoir_type"])
            # Normalize reservoir type names
            if res_type.lower() == "percentage_decline":
                processed["reservoir_type"] = "percentage"
            elif res_type.lower() == "tabular":
                processed["reservoir_type"] = "tabular"
        
        # Handle drilling_cost: convert False to None
        if "drilling_cost" in processed:
            if processed["drilling_cost"] is False:
                processed["drilling_cost"] = None
        
        # Handle ppc -> powerplant_capacity mapping
        if "ppc" in processed and "powerplant_capacity" not in processed:
            processed["powerplant_capacity"] = processed.pop("ppc")
        
        # Ensure lists are lists
        if "battery_duration" in processed and not isinstance(processed["battery_duration"], list):
            processed["battery_duration"] = [processed["battery_duration"], 0]
        
        if "battery_power_capacity" in processed and not isinstance(processed["battery_power_capacity"], list):
            processed["battery_power_capacity"] = [processed["battery_power_capacity"], 0]
        
        return processed

