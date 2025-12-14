"""Default configuration values for FGEM.

All default values are organized by category using dataclasses.
This ensures no hardcoded values exist in the business logic.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import os


@dataclass
class MetadataDefaults:
    """Default metadata and project settings."""
    base_dir: str = os.path.join(os.getcwd(), "./data/")
    data_dir: str = "Data"
    time_init: str = "2025-01-01"
    market_dir: str = "market"
    state: Optional[str] = None
    project_lat: Optional[float] = None
    project_long: Optional[float] = None
    n_jobs: int = 1
    reservoir_simulator_settings: Dict[str, Any] = field(default_factory=lambda: {
        "fast_mode": True,
        "period": 2628000,
        "accuracy": 1,
        "DynamicFluidProperties": True
    })


@dataclass
class EconomicsDefaults:
    """Default economics parameters."""
    L: int = 30  # Project lifetime in years
    d: float = 0.07  # Discount rate
    itc: float = 0.0  # Investment tax credit
    inflation: float = 0.02
    contingency: float = 0.15
    opex_escalation: float = 0.0
    baseline_year: int = 2025
    # Drilling cost user-facing default:
    # - Default: 10.0 MM$ per well (matches scalable EGS example behavior when not specified)
    # - If value <= 200, ConfigManager interprets it as MM$ per well and converts to USD/m internally.
    # - If value > 200, ConfigManager interprets it as USD/m (backward compatible with existing examples).
    # - Calculation method is consistent across all reservoir types (same compute_drilling_cost function).
    #   Only well geometry differs: uloop uses half_lateral_length, others use lateral_length.
    drilling_cost: Optional[float] = 10.0
    powerplant_interconnection_cost: float = 130  # USD/kW
    exploration_cost_intercept: float = 1.0  # $MM
    exploration_cost_slope: float = 0.6  # USD exploration / USD producer capex
    stimulation_cost: float = 2.5  # $MM per injection well


@dataclass
class BatteryDefaults:
    """Default battery storage parameters."""
    battery_costs_filename: Optional[str] = None
    battery_duration: List[float] = field(default_factory=lambda: [0, 0])
    battery_power_capacity: List[float] = field(default_factory=lambda: [0, 0])
    battery_interconnection_cost: float = 200  # USD/kW
    battery_energy_cost: float = 200  # USD/kWh
    battery_power_cost: float = 300  # USD/kW
    battery_fom: float = 10  # Fixed O&M $/kW-year
    battery_energy_augmentation: float = 3  # $/kWh-year
    battery_power_augmentation: float = 0.5  # $/kW-year
    battery_elcc: float = 1.0  # Effective Load Carrying Capability
    battery_roundtrip_eff: float = 0.85
    battery_lifetime: int = 15  # years


@dataclass
class ThermalStorageDefaults:
    """Default thermal energy storage (TES) parameters."""
    tank_diameter: float = 0  # meters
    tank_height: float = 0  # meters
    tank_cost: float = 0.00143453  # $/m³


@dataclass
class PowerPlantDefaults:
    """Default power plant parameters."""
    powerplant_capacity: Optional[float] = None  # MWe
    powerplant_type: str = "Binary"
    pipinglength: float = 5  # km
    cf: float = 1.0  # Capacity factor
    bypass: bool = False
    powerplant_interconnection_cost: float = 130  # USD/kW
    powerplant_opex_rate: float = 0.015  # 1.5/100
    powerplant_labor_rate: float = 0.75
    powerplant_usd_per_kw_min: float = 2000  # USD/kWe
    Tres_pp_design: Optional[float] = None
    m_prd_pp_design: float = 80  # kg/s
    powerplant_k: int = 2


@dataclass
class WellsiteDefaults:
    """Default wellsite operation parameters."""
    wellsite_opex_rate: float = 0.01  # 1/100
    wellsite_labor_rate: float = 0.25


@dataclass
class MarketDefaults:
    """Default power market parameters."""
    energy_price: float = 40  # $/MWh (if not using file)
    energy_market_filename: Optional[str] = None
    capacity_price: float = 100  # $/MW-hour (if not using file)
    capacity_market_filename: Optional[str] = None
    recs_price: float = 10  # $/MWh (Renewable Energy Credits)
    ppa_price: float = 70  # $/MWh
    ppa_escalaction_rate: float = 0.02
    fat_factor: float = 1
    resample: str = "1d"
    oversample_first_day: Optional[int] = None


@dataclass
class WeatherDefaults:
    """Default weather parameters."""
    weather_filename: Optional[str] = None
    surface_temp: float = 20  # °C
    sup3rcc_weather_forecast: bool = False


@dataclass
class ReservoirDefaults:
    """Default reservoir parameters."""
    reservoir_filename: Optional[str] = None
    reservoir_type: str = "diffusion_convection"
    Pres_init: float = 40  # bar
    V_res: float = 5  # km³
    phi_res: float = 0.1  # porosity
    res_thickness: float = 1000  # meters
    krock: float = 30  # W/m-K (thermal conductivity)
    cprock: float = 1100  # J/kg-K (specific heat)
    drawdp: float = 0.005  # drawdown percentage
    plateau_length: float = 3  # years
    rock_energy_recovery: float = 1.0
    geothermal_gradient: float = 65  # °C/km
    Tres_init: Optional[float] = None  # Will be calculated if None
    redrill_ratio: Optional[float] = None
    shutoff_ratio: Optional[float] = None
    total_drilling_length: Optional[float] = None
    prd_total_drilling_length: Optional[float] = None
    inj_total_drilling_length: Optional[float] = None


@dataclass
class WellDefaults:
    """Default well parameters."""
    well_tvd: float = 3500  # meters (True Vertical Depth)
    well_md: Optional[float] = None  # meters (Measured Depth)
    prd_well_diam: float = 0.3115  # meters
    inj_well_diam: float = 0.3115  # meters
    int_well_diam: float = 0.3115  # meters
    lateral_length: float = 2000  # meters (default 2000, set to 0 for vertical wells)
    lateral_diam: float = 0.3115  # meters
    lateral_spacing: float = 100  # meters
    numberoflaterals: int = 1
    num_prd: int = 4
    inj_prd_ratio: float = 1.0
    waterloss: float = 0.0  # fraction
    pumpeff: float = 0.8  # pump efficiency
    DSR: float = 1.0  # Drilling Success Rate
    SSR: float = 1.0  # Stimulation Success Rate
    PI: float = 8  # Productivity Index (l/s/bar)
    II: float = 8  # Injectivity Index (l/s/bar)
    PumpingModel: str = "OpenLoop"
    closedloop_design: str = "default"
    dx: Optional[float] = None
    ramey: Optional[bool] = None
    m_prd: float = 110  # kg/s


@dataclass
class CoaxialDefaults:
    """Default coaxial well parameters."""
    casing_inner_diam: float = 0.13208  # meters
    tube_inner_diam: float = 0.0620014  # meters
    tube_thickness: float = 0.0197993  # meters (0.0395986/2)
    k_tube: float = 0.088  # W/m-K (thermal conductivity)
    coaxialflowtype: int = 1  # 1=CXA, 2=CXC


@dataclass
class SimulatorSettingsDefaults:
    """Default simulator settings for reservoir models."""
    # General settings
    ramey: bool = True
    pumping: bool = True
    timestep_hours: float = 1.0  # Simulation timestep in hours
    
    # Rock properties (BaseReservoir defaults)
    krock: float = 3.0  # W/m-K (thermal conductivity)
    rhorock: float = 2700.0  # kg/m³ (bulk density)
    cprock: float = 1000.0  # J/kg-K (specific heat)
    krock_wellbore: float = 3.0  # W/m-K (thermal conductivity around wellbore)
    
    # Reservoir properties
    impedance: float = 0.1  # Reservoir pressure losses
    N_ramey_mv_avg: int = 168  # Timesteps for averaging Ramey's f-function
    
    # ULoop specific settings
    k_m: float = 2.83  # Rock thermal conductivity for ULoop (W/m-K)
    rho_m: float = 2875.0  # Rock bulk density for ULoop (kg/m³)
    c_m: float = 825.0  # Rock heat capacity for ULoop (J/kg-K)
    k_f: float = 0.68  # Fluid thermal conductivity (W/m-K)
    mu_f: float = 6e-4  # Fluid kinematic viscosity (m²/s) = 600e-6
    cp_f: float = 4200.0  # Fluid heat capacity (J/kg-K)
    rho_f: float = 1000.0  # Fluid density (kg/m³)
    fullyimplicit: int = 1  # Euler's method solver option
    FMM: int = 1  # Fast multi-pole method (1=enabled)
    FMMtriggertime: float = 864000.0  # FMM trigger time in seconds (3600*24*10)
    
    # Coaxial specific settings (already in CoaxialDefaults, but included for completeness)
    # coaxialflowtype is in CoaxialDefaults
    
    # DiffusionConvection specific overrides (different from base)
    krock_diffusion: float = 30.0  # W/m-K for DiffusionConvection
    rhorock_diffusion: float = 2600.0  # kg/m³ for DiffusionConvection
    cprock_diffusion: float = 1100.0  # J/kg-K for DiffusionConvection


@dataclass
class ConfigDefaults:
    """Complete configuration defaults container."""
    metadata: MetadataDefaults = field(default_factory=MetadataDefaults)
    economics: EconomicsDefaults = field(default_factory=EconomicsDefaults)
    battery: BatteryDefaults = field(default_factory=BatteryDefaults)
    thermal_storage: ThermalStorageDefaults = field(default_factory=ThermalStorageDefaults)
    powerplant: PowerPlantDefaults = field(default_factory=PowerPlantDefaults)
    wellsite: WellsiteDefaults = field(default_factory=WellsiteDefaults)
    market: MarketDefaults = field(default_factory=MarketDefaults)
    weather: WeatherDefaults = field(default_factory=WeatherDefaults)
    reservoir: ReservoirDefaults = field(default_factory=ReservoirDefaults)
    well: WellDefaults = field(default_factory=WellDefaults)
    coaxial: CoaxialDefaults = field(default_factory=CoaxialDefaults)
    simulator_settings: SimulatorSettingsDefaults = field(default_factory=SimulatorSettingsDefaults)

    def to_dict(self) -> dict:
        """Convert defaults to flat dictionary format compatible with World class."""
        config = {}
        
        # Metadata
        config.update({
            "project_data_dir": self.metadata.base_dir,
            "n_jobs": self.metadata.n_jobs,
            "time_init": self.metadata.time_init,
            "project_lat": self.metadata.project_lat,
            "project_long": self.metadata.project_long,
            "reservoir_simulator_settings": self.metadata.reservoir_simulator_settings,
        })
        
        # Economics
        config.update({
            "L": self.economics.L,
            "d": self.economics.d,
            "itc": self.economics.itc,
            "inflation": self.economics.inflation,
            "contingency": self.economics.contingency,
            "drilling_cost": self.economics.drilling_cost,
            "powerplant_interconnection_cost": self.economics.powerplant_interconnection_cost,
            "exploration_cost_intercept": self.economics.exploration_cost_intercept,
            "exploration_cost_slope": self.economics.exploration_cost_slope,
            "stimulation_cost": self.economics.stimulation_cost,
        })
        
        # Battery
        config.update({
            "battery_costs_filename": self.battery.battery_costs_filename,
            "battery_duration": self.battery.battery_duration,
            "battery_power_capacity": self.battery.battery_power_capacity,
            "battery_interconnection_cost": self.battery.battery_interconnection_cost,
            "battery_energy_cost": self.battery.battery_energy_cost,
            "battery_power_cost": self.battery.battery_power_cost,
            "battery_fom": self.battery.battery_fom,
            "battery_energy_augmentation": self.battery.battery_energy_augmentation,
            "battery_power_augmentation": self.battery.battery_power_augmentation,
            "battery_elcc": self.battery.battery_elcc,
            "battery_roundtrip_eff": self.battery.battery_roundtrip_eff,
            "battery_lifetime": self.battery.battery_lifetime,
        })
        
        # Thermal Storage
        config.update({
            "tank_diameter": self.thermal_storage.tank_diameter,
            "tank_height": self.thermal_storage.tank_height,
            "tank_cost": self.thermal_storage.tank_cost,
        })
        
        # Power Plant
        config.update({
            "powerplant_capacity": self.powerplant.powerplant_capacity,
            "powerplant_type": self.powerplant.powerplant_type,
            "pipinglength": self.powerplant.pipinglength,
            "cf": self.powerplant.cf,
            "bypass": self.powerplant.bypass,
            "powerplant_opex_rate": self.powerplant.powerplant_opex_rate,
            "powerplant_labor_rate": self.powerplant.powerplant_labor_rate,
            "powerplant_usd_per_kw_min": self.powerplant.powerplant_usd_per_kw_min,
            "Tres_pp_design": self.powerplant.Tres_pp_design,
            "m_prd_pp_design": self.powerplant.m_prd_pp_design,
            "powerplant_k": self.powerplant.powerplant_k,
        })
        
        # Wellsite
        config.update({
            "wellsite_opex_rate": self.wellsite.wellsite_opex_rate,
            "wellsite_labor_rate": self.wellsite.wellsite_labor_rate,
        })
        
        # Market
        config.update({
            "energy_price": self.market.energy_price,
            "energy_market_filename": self.market.energy_market_filename,
            "capacity_price": self.market.capacity_price,
            "capacity_market_filename": self.market.capacity_market_filename,
            "recs_price": self.market.recs_price,
            "ppa_price": self.market.ppa_price,
            "ppa_escalaction_rate": self.market.ppa_escalaction_rate,
            "fat_factor": self.market.fat_factor,
            "resample": self.market.resample,
            "oversample_first_day": self.market.oversample_first_day,
        })
        
        # Weather
        config.update({
            "weather_filename": self.weather.weather_filename,
            "surface_temp": self.weather.surface_temp,
            "sup3rcc_weather_forecast": self.weather.sup3rcc_weather_forecast,
        })
        
        # Reservoir
        config.update({
            "reservoir_filename": self.reservoir.reservoir_filename,
            "reservoir_type": self.reservoir.reservoir_type,
            "Pres_init": self.reservoir.Pres_init,
            "V_res": self.reservoir.V_res,
            "phi_res": self.reservoir.phi_res,
            "res_thickness": self.reservoir.res_thickness,
            "krock": self.reservoir.krock,
            "cprock": self.reservoir.cprock,
            "drawdp": self.reservoir.drawdp,
            "plateau_length": self.reservoir.plateau_length,
            "rock_energy_recovery": self.reservoir.rock_energy_recovery,
            "geothermal_gradient": self.reservoir.geothermal_gradient,
            "Tres_init": self.reservoir.Tres_init,
            "redrill_ratio": self.reservoir.redrill_ratio,
            "shutoff_ratio": self.reservoir.shutoff_ratio,
            "total_drilling_length": self.reservoir.total_drilling_length,
            "prd_total_drilling_length": self.reservoir.prd_total_drilling_length,
            "inj_total_drilling_length": self.reservoir.inj_total_drilling_length,
        })
        
        # Well
        config.update({
            "well_tvd": self.well.well_tvd,
            "well_md": self.well.well_md,
            "prd_well_diam": self.well.prd_well_diam,
            "inj_well_diam": self.well.inj_well_diam,
            "int_well_diam": self.well.int_well_diam,
            "lateral_length": self.well.lateral_length,
            "lateral_diam": self.well.lateral_diam,
            "lateral_spacing": self.well.lateral_spacing,
            "numberoflaterals": self.well.numberoflaterals,
            "num_prd": self.well.num_prd,
            "inj_prd_ratio": self.well.inj_prd_ratio,
            "waterloss": self.well.waterloss,
            "pumpeff": self.well.pumpeff,
            "DSR": self.well.DSR,
            "SSR": self.well.SSR,
            "PI": self.well.PI,
            "II": self.well.II,
            "PumpingModel": self.well.PumpingModel,
            "closedloop_design": self.well.closedloop_design,
            "dx": self.well.dx,
            "ramey": self.well.ramey,
            "m_prd": self.well.m_prd,
        })
        
        # Coaxial
        config.update({
            "casing_inner_diam": self.coaxial.casing_inner_diam,
            "tube_inner_diam": self.coaxial.tube_inner_diam,
            "tube_thickness": self.coaxial.tube_thickness,
            "k_tube": self.coaxial.k_tube,
            "coaxialflowtype": self.coaxial.coaxialflowtype,
        })
        
        # Simulator Settings
        sim = self.simulator_settings
        config.update({
            "ramey": sim.ramey,
            "pumping": sim.pumping,
            "timestep_hours": sim.timestep_hours,
            "krock_wellbore": sim.krock_wellbore,
            "impedance": sim.impedance,
            "N_ramey_mv_avg": sim.N_ramey_mv_avg,
            # ULoop settings
            "k_m": sim.k_m,
            "rho_m": sim.rho_m,
            "c_m": sim.c_m,
            "k_f": sim.k_f,
            "mu_f": sim.mu_f,
            "cp_f": sim.cp_f,
            "rho_f": sim.rho_f,
            "fullyimplicit": sim.fullyimplicit,
            "FMM": sim.FMM,
            "FMMtriggertime": sim.FMMtriggertime,
            # DiffusionConvection overrides (will be used conditionally)
            "krock_diffusion": sim.krock_diffusion,
            "rhorock_diffusion": sim.rhorock_diffusion,
            "cprock_diffusion": sim.cprock_diffusion,
        })
        
        return config


def get_defaults() -> ConfigDefaults:
    """Get default configuration values."""
    return ConfigDefaults()

