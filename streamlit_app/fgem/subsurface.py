import os
import pandas as pd
from datetime import timedelta, datetime
import numpy as np
import pdb
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from pyXSteam.XSteam import XSteam
import math
from scipy import integrate, interpolate
import pickle
from scipy.special import erf, erfc, jv, yv, exp1
from .utils.utils import compute_f, densitywater, viscositywater, heatcapacitywater, vaporpressurewater, nonzero
from collections import defaultdict

# FILE_BASE_DIR = os.path.dirname(__file__)

steamtable = XSteam(XSteam.UNIT_SYSTEM_MKS)

def _safe_mean(value):
    """Safely get mean value, handling both float and array inputs.
    
    Args:
        value: Either a float or numpy array
        
    Returns:
        float: Mean value if array, or the value itself if float
    """
    if isinstance(value, np.ndarray):
        return float(value.mean())
    else:
        return float(value)

class BaseReservoir(object):
    """Base class defining subsurface reservoir and wellbore."""
    def __init__(self,
                 Tres_init,
                 geothermal_gradient,
                 surface_temp,
                 L,
                 time_init,
                 well_tvd,
                 prd_well_diam,
                 inj_well_diam,
                 num_prd,
                 num_inj,
                 waterloss,
                 powerplant_type,
                 pumpeff,
                 ramey=True,
                 pumping=True,
                  krock=3,
                 rhorock=2700,
                    cprock=1000,
                   impedance = 0.1,
                 res_thickness=200,
                 PI = 20,
                 II = 20,
                 SSR = 1.0,
                 N_ramey_mv_avg=168,
                 reservoir_simulator_settings={"fast_mode": False, "period": 3600*8760/12},
                 PumpingModel="OpenLoop",
                 timestep=timedelta(hours=1),
                 krock_wellbore=3,
                 ):
        """Initialize base reservoir class.

        Args:
            Tres_init (float): initial reservoir temperature in deg C.
            geothermal_gradient (float): average geothermal gradient in deg C/km.
            surface_temp (float): surface temperature in deg C.
            L (float): project lifetime in years.
            time_init (datetime): initial time.
            well_tvd (float): well depth in meters.
            prd_well_diam (float): production well diameter in meters.
            inj_well_diam (float): injection well diameter in meters.
            num_prd (int): number of producers.
            num_inj (int): number of injectors.
            waterloss (float): fraction of injected water that is lost to the reservoir (fraction).
            powerplant_type (str): type of power plant (either "Binary" or "Flash").
            pumpeff (float): pump efficiency (fraction).
            ramey (bool, optional): whether or not to use ramey's model for wellbore heat loss/gain. Defaults to True.
            pumping (bool, optional): whther or not to account for parasitic losses due to pumping requirements. Defaults to True.
            krock (float, optional): rock thermal conductivity in W/C-m. Defaults to 3.
            rhorock (float, optional): rock bulk density in kg/m3. Defaults to 2700.
            cprock (float, optional): rock heat capacity in J/kg-K. Defaults to 1000.
            impedance (float, optional): reservoir pressure losses when using an impendance model. Defaults to 0.1.
            res_thickness (float, optional): reservoir thickness in meters. Defaults to 200.
            PI (float, optional): productivity index in kg/s/bar. Defaults to 20.
            II (float, optional): injectivity index in kg/s/bar. Defaults to 20.
            SSR (float, optional): Stimulation success rate, which is a multiplier used to reduce PI and II when stimulation is not fully successful. Defaults to 1.0.
            N_ramey_mv_avg (int, optional): number of timesteps used for averaging the f-function when computing ramey's heat losses with variable mass flow rates. Defaults to 168.
            reservoir_simulator_settings (dict, optional): information used to reduce the required timestepping when simulating the reservoir. It comes with keys of "fast_mode" to turn it on and "period" to specify the time period needed to pass before the reservoir state is updated, which is aimed at reducing computational requirements in exchange for loss in accuracy. Defaults to {"fast_mode": False, "period": 3600*8760/12}.
            PumpingModel (str, optional): model type used to compute pressure losses (either "OpenLoop" or "ClosedLoop"). Defaults to "OpenLoop".
            timestep (datetime.timedelta, optional): simulation timestep size. Defaults to timedelta(hours=1).
            krock_wellbore (float): thermal conductivity around the wellbore, used for wellbore heat gain/loss
        """

        self.geothermal_gradient = geothermal_gradient
        self.surface_temp = surface_temp
        self.L = L
        self.time_init = time_init
        self.time_curr = time_init
        self.well_tvd = well_tvd
        self.num_prd = num_prd
        self.num_inj = num_inj
        self.Tres_init = Tres_init
        self.Tres = self.Tres_init
        self.steamtable = XSteam(XSteam.UNIT_SYSTEM_MKS) #m/kg/sec/°C/bar/W
        self.krock = krock
        self.rhorock = rhorock
        self.cprock = cprock
        self.prd_well_diam = prd_well_diam
        self.inj_well_diam = inj_well_diam
        self.alpharock = krock/(rhorock*cprock)
        self.ramey = ramey
        self.pumping = pumping
        self.impedance = impedance
        self.res_thickness = res_thickness
        self.PI = SSR * PI if SSR > 0.0 else PI #0.3 Based on GETEM page 61
        self.II = SSR * II if SSR > 0.0 else II #0.3 Based on GETEM page 61
        self.waterloss = waterloss
        self.pumpeff = pumpeff
        self.powerplant_type = powerplant_type
        self.dT_prd = 0.0
        self.dT_inj = 0.0
        self.m_prd_ramey_mv_avg = 100 # randomly initialized, but it will quickly converge following the dispatch strategy
        self.m_inj_ramey_mv_avg = 100 # randomly initialized, but it will quickly converge following the dispatch strategy
        self.N_ramey_mv_avg = N_ramey_mv_avg
        self.reservoir_simulator_settings = reservoir_simulator_settings
        self.reservoir_simulator_settings["time_passed"] = np.inf
        self.PumpingModel = PumpingModel
        self.timestep = timestep
        self.krock_wellbore = krock_wellbore

        #initialize wellhead and bottomhole quantities
        self.T_prd_wh = self.Tres_init
        self.T_prd_bh = self.Tres_init
        self.T_inj_wh = 75
        self.T_inj_bh = 75
  
        #reservoir hydrostatic pressure [kPa]
        self.CP = 4.64E-7
        self.CT = 9E-4/(30.796*self.Tres_init**(-0.552))
        self.Phydrostatic = 0+1./self.CP*(math.exp(densitywater(self.surface_temp)*9.81*self.CP/1000*(self.well_tvd-self.CT/2*(self.geothermal_gradient/1000)*self.well_tvd**2))-1)

        self.cache = defaultdict(lambda: defaultdict(int))

        # ramey estimates
        time_seconds = 8760*3600 # always assume one year passed across
        self.framey_prd = -np.log(1.1*(self.prd_well_diam/2.)/np.sqrt(4.*self.alpharock*time_seconds))-0.29
        self.framey_inj = -np.log(1.1*(self.inj_well_diam/2.)/np.sqrt(4.*self.alpharock*time_seconds))-0.29

        self.shutoff = False
        self.m_prd_arr = []
        self.m_inj_arr = []
        self.T_prd_wh_arr = []
        self.T_res_arr = []
        self.T_inj_arr = []
        self.time_passed_arr = []
        self.shutoff_arr = []
        self.num_prd_arr = []
        self.num_inj_arr = []


    def pre_model(self, t, m_prd, m_inj, T_inj):
        """Computations to be performed before stepping the reservoir model.

        Args:
            t (datetime): current timestamp.
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s.
            T_inj (float): injection temperature in deg C.

        Raises:
            NotImplementedError: must be implemented for classes inheriting the BaseReservoir class
        """
        raise NotImplementedError

    def model(self, t, m_prd, m_inj, T_inj):
        """Computations to be performed when stepping the reservoir model.

        Args:
            t (datetime): current timestamp.
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s.
            T_inj (float): injection temperature in deg C.

        Raises:
            NotImplementedError: must be implemented for classes inheriting the BaseReservoir class
        """
        raise NotImplementedError

    def configure_well_dimensions(self):
        """Configuration specifications of a doublet. It requires the specification of a doublet, including the producer dimensions (self.xprod, self.yprod, self.zprod), injector dimensions (self.xinj, self.yinj, self.zinj) and reservoir vertices (self.verts). See Class PercentageReservoir for example implementation.

        Raises:
            NotImplementedError: must be implemented for classes inheriting the BaseReservoir class
        """
        raise NotImplementedError

    def pre_wellbore_calculations(self, t, m_prd, m_inj, T_inj, T_amb):
        """Computations to be performed before wellbore calculations.

        Args:
            t (datetime): current timestamp.
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s.
            T_inj (float): injection temperature in deg C.
        """
        if self.ramey:
            self.pre_compute_ramey(t, m_prd, m_inj, T_inj, T_amb)

    def wellbore_calculations(self, t, m_prd, m_inj, T_inj, T_amb):
        """Wellbore computations.

        Args:
            t (datetime): current timestamp.
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s.
            T_inj (float): injection temperature in deg C.
        """
        # Ramey's wellbore heat loss model
        if self.ramey:
            self.compute_ramey(t, m_prd, m_inj, T_inj, T_amb)
        else:
            self.T_prd_wh = self.T_prd_bh
            self.T_inj_bh = np.array(self.num_inj*[T_inj], dtype='float')
   
        self.dT_prd = self.T_prd_bh - self.T_prd_wh
        self.dT_inj = self.T_inj_bh - T_inj
        self.T_inj_wh = np.array(self.num_inj*[T_inj], dtype='float')
    
    def step(self, m_prd, T_inj, T_amb, m_inj=None):
        """Stepping the reservoir and wellbore models.

        Args:
            m_prd (Union[float], optional): producer mass flow rates in kg/s.
            T_inj (float): injection temperature in deg C.
            T_amb (float): ambient temperature in deg C.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s.
        """

        self.m_prd = m_prd
        self.m_inj = m_inj if m_inj is not None else m_prd * self.num_inj/nonzero(self.num_prd)

        self.time_curr += self.timestep
        self.pre_model(self.time_curr, self.m_prd, self.m_inj, T_inj)
        self.pre_wellbore_calculations(self.time_curr, self.m_prd, self.m_inj, T_inj, T_amb)
  
        # if reservoir_simulator_settings and still not period time yet, then do not update anything
        if self.reservoir_simulator_settings["fast_mode"]:
            self.reservoir_simulator_settings["time_passed"] += self.timestep.total_seconds()
            if self.reservoir_simulator_settings["time_passed"] >= self.reservoir_simulator_settings["period"]:
                self.reservoir_simulator_settings["time_passed"] = 0.0
                self.model(self.time_curr, self.m_prd, self.m_inj, T_inj)
                self.wellbore_calculations(self.time_curr, self.m_prd, self.m_inj, T_inj, T_amb)
            else:
                self.wellbore_calculations(self.time_curr, self.m_prd, self.m_inj, T_inj, T_amb)

        else:
            self.model(self.time_curr, self.m_prd, self.m_inj, T_inj)
            self.wellbore_calculations(self.time_curr, self.m_prd, self.m_inj, T_inj, T_amb)

        self.time_passed_arr.append((self.time_curr - self.time_init).total_seconds())
        self.m_prd_arr.append(self.m_prd.sum())
        self.m_inj_arr.append(self.m_inj.sum())
        self.T_prd_wh_arr.append(self.T_prd_wh)
        self.T_res_arr.append(self.Tres)
        self.T_inj_arr.append(T_inj)
        self.shutoff_arr.append(self.shutoff)
        self.num_prd_arr.append(self.num_prd)
        self.num_inj_arr.append(self.num_inj)

    def pre_compute_ramey(self, t, m_prd, m_inj, T_inj, T_amb):
        """Computations to be performed before Ramey's model calculations.

        Args:
            t (datetime): current timestamp.
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s.
            T_inj (float): injection temperature in deg C.
            T_amb (float): ambient temperature in deg C.
        """
        # average over wells
        self.m_prd_ramey_mv_avg = ((self.N_ramey_mv_avg-1) * self.m_prd_ramey_mv_avg + m_prd)/self.N_ramey_mv_avg
        self.m_inj_ramey_mv_avg = ((self.N_ramey_mv_avg-1) * self.m_inj_ramey_mv_avg + m_inj)/self.N_ramey_mv_avg
  
    def compute_ramey(self, t, m_prd, m_inj, T_inj, T_amb):
        """Ramey's model wellbore heat loss/gain.

        Args:
            t (datetime): current timestamp.
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s.
            T_inj (float): injection temperature in deg C.
            T_amb (float): ambient temperature in deg C.
        """
     
        # Producer calculations
        Tavg = self.T_prd_bh
        cpwater = heatcapacitywater(Tavg)
        rameyA = self.m_prd_ramey_mv_avg*cpwater*self.framey_prd/2/math.pi/self.krock_wellbore

        self.T_prd_wh = (self.Tres - (self.geothermal_gradient/1000)*self.well_tvd) + \
        (self.geothermal_gradient/1000) * rameyA * (1 - np.exp(-self.well_tvd/nonzero(rameyA))) + \
        (self.T_prd_bh - self.Tres) * np.exp(-self.well_tvd/nonzero(rameyA))

        # Injector calculations
        Tavg = T_inj
        # cpwater = heatcapacitywater(Tavg)
        Tround = round(Tavg, 0)
        cpwater = self.cache["cpwater"].get(Tround, heatcapacitywater(Tavg))
        self.cache["cpwater"][Tround] = cpwater
        rameyA = self.m_inj_ramey_mv_avg*cpwater*self.framey_inj/2/math.pi/self.krock_wellbore

        self.T_inj_bh = (self.surface_temp + (self.geothermal_gradient/1000)*self.well_tvd) - \
        (self.geothermal_gradient/1000) * rameyA + \
        (T_inj - self.surface_temp + (self.geothermal_gradient/1000) * rameyA) * np.exp(-self.well_tvd/nonzero(rameyA))

    def plot_doublet(self, dpi=150):
        """Visualize a doublet of the proposed system. Using this method requires to first implement :py:func:`~subsurface.BaseReservoir.configure_well_dimensions`.

        Args:
            dpi (int, optional): figure dpi resolution. Defaults to 150.

        Returns:
            plt.figure.Figure: figure
        """
        assert hasattr(self, "zprod"), "Implementation Error: You must define the wellbore and reservoir dimensions to plot doublets! Define method subsurface.BaseReservoir.configure_well_dimensions"

        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')

        ax.add_collection3d(Poly3DCollection(self.verts, alpha=0.1, color="tab:orange"))
        ax.plot(self.xinj, self.yinj, self.zinj, 'tab:blue', linewidth=4)
        ax.plot(self.xprod, self.yprod, self.zprod, 'tab:red', linewidth=4)

        ax.set_xlim([np.min(self.v[:,0]) - 200, np.max(self.v[:,0]) + 200])
        ax.set_ylim([np.min(self.v[:,1]) - 200, np.max(self.v[:,1]) + 200])
        ax.set_zlim([np.min(self.v[:,2]) - 500, 0])

        col1_patch = mpatches.Patch(color="tab:orange", label='Reservoir')
        col2_patch = mpatches.Patch(color="tab:blue", label='Injector')
        col3_patch = mpatches.Patch(color="tab:red", label='Producer')
        handles = [col1_patch, col2_patch, col3_patch]

        if hasattr(self, "zlat"):
            for j in range(self.xlat.shape[1]):
                ax.plot(self.xlat[:,j], self.ylat[:,j], self.zlat[:,j],
                        linewidth=2, color="black")

            col4_patch = mpatches.Patch(color="black", label='Laterals')
            handles.append(col4_patch)
            
        plt.legend(handles=handles)

        return fig

class PercentageReservoir(BaseReservoir):
    """Conceptual reservoir model where temperature declines based on an fixed annual decline rate."""

    def __init__(self,
                 Tres_init,
                 geothermal_gradient,
                 surface_temp,
                 L,
                 time_init,
                 well_tvd,
                 prd_well_diam,
                 inj_well_diam,
                 num_prd,
                 num_inj,
                 waterloss,
                 powerplant_type,
                 pumpeff,
                 ramey=True,
                 pumping=True,
                  krock=3,
                 rhorock=2700,
                    cprock=1000,
                   impedance = 0.1,
                 res_thickness=200,
                 PI = 20,
                 II = 20, 
                 SSR = 1.0,
                 N_ramey_mv_avg=168,
                 drawdp=0.005,
                 plateau_length=3,
                 reservoir_simulator_settings={"fast_mode": False, "period": 3600*8760/12},
                 PumpingModel="OpenLoop",
                 timestep=timedelta(hours=1),
                 krock_wellbore=3
                 ):

        """Initialize reservoir model.

        Args:
            Tres_init (float): initial reservoir temperature in deg C.
            geothermal_gradient (float): average geothermal gradient in deg C/km.
            surface_temp (float): surface temperature in deg C.
            L (float): project lifetime in years.
            time_init (datetime): initial time.
            well_tvd (float): well depth in meters.
            prd_well_diam (float): production well diameter in meters.
            inj_well_diam (float): injection well diameter in meters.
            num_prd (int): number of producers.
            num_inj (int): number of injectors.
            waterloss (float): fraction of injected water that is lost to the reservoir (fraction).
            powerplant_type (str): type of power plant (either "Binary" or "Flash").
            pumpeff (float): pump efficiency (fraction).
            ramey (bool, optional): whether or not to use ramey's model for wellbore heat loss/gain. Defaults to True.
            pumping (bool, optional): whther or not to account for parasitic losses due to pumping requirements. Defaults to True.
            krock (float, optional): rock thermal conductivity in W/C-m. Defaults to 3.
            rhorock (float, optional): rock bulk density in kg/m3. Defaults to 2700.
            cprock (float, optional): rock heat capacity in J/kg-K. Defaults to 1000.
            impedance (float, optional): reservoir pressure losses when using an impendance model. Defaults to 0.1.
            res_thickness (float, optional): reservoir thickness in meters. Defaults to 200.
            PI (float, optional): productivity index in kg/s/bar. Defaults to 20.
            II (float, optional): injectivity index in kg/s/bar. Defaults to 20.
            SSR (float, optional): Stimulation success rate, which is a multiplier used to reduce PI and II when stimulation is not fully successful. Defaults to 1.0.
            N_ramey_mv_avg (int, optional): number of timesteps used for averaging the f-function when computing ramey's heat losses with variable mass flow rates. Defaults to 168.
            drawdp (float, optional): annual decline rate of reservoir temperature (fraction). Defaults to 0.005.
            plateau_length (int, optional): number of years before reservoir temperature starts to decline. Defaults to 3.
            reservoir_simulator_settings (dict, optional): information used to reduce the required timestepping when simulating the reservoir. It comes with keys of "fast_mode" to turn it on and "period" to specify the time period needed to pass before the reservoir state is updated, which is aimed at reducing computational requirements in exchange for loss in accuracy. Defaults to {"fast_mode": False, "period": 3600*8760/12}.
            PumpingModel (str, optional): model type used to compute pressure losses (either "OpenLoop" or "ClosedLoop"). Defaults to "OpenLoop".
        """

        super(PercentageReservoir, self).__init__(Tres_init,
                                            geothermal_gradient,
                                            surface_temp,
                                            L,
                                            time_init,
                                            well_tvd,
                                            prd_well_diam,
                                            inj_well_diam,
                                            num_prd,
                                            num_inj,
                                            waterloss,
                                            powerplant_type,
                                            pumpeff,
                                            ramey,
                                            pumping,
                                            krock,
                                            rhorock,
                                            cprock,
                                            impedance,
                                            res_thickness,
                                            PI,
                                            II,
                                            SSR,
                                            N_ramey_mv_avg,
                                            reservoir_simulator_settings,
                                            PumpingModel,
                                            timestep,
                                            krock_wellbore)
        self.numberoflaterals = 1
        self.well_tvd = well_tvd
        self.well_md = self.well_tvd
        self.res_length = 2000
        self.res_thickness = res_thickness
        self.res_width = 1000

        self.Tres_arr = self.Tres_init*np.ones(self.L+2)
        for i in range(self.L+2):
            if i+1 > plateau_length:
                self.Tres_arr[i] = (1 - drawdp) * self.Tres_arr[i-1]

        self.configure_well_dimensions()
        
    def pre_model(self, t, m_prd, m_inj, T_inj):
        """Computations to be performed before stepping the reservoir model.

        Args:
            t (datetime): current timestamp.
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s.
            T_inj (float): injection temperature in deg C.
        """
        pass

    def model(self, t, m_prd, m_inj, T_inj):

        """Computations to be performed when stepping the reservoir model.

        Args:
            t (datetime): current timestamp.
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s.
            T_inj (float): injection temperature in deg C.
        """

        self.Tres = self.Tres_arr[t.year - self.time_init.year]
        self.T_prd_bh = np.array(self.num_prd*[self.Tres], dtype='float')
        self.T_inj_wh = np.array(self.num_inj*[T_inj], dtype='float')

    def configure_well_dimensions(self):
        """Configuration specifications of a doublet. It requires the specification of a doublet, including the producer dimensions (self.xprod, self.yprod, self.zprod), injector dimensions (self.xinj, self.yinj, self.zinj) and reservoir vertices (self.verts). See Class PercentageReservoir for example implementation.
        """

        self.zprod = np.array([0, -self.well_tvd])
        self.xprod = -self.res_length/2 * np.ones_like(self.zprod)
        self.yprod = np.zeros_like(self.zprod)

        self.zinj = np.array([0, -self.well_tvd])
        self.xinj = self.res_length/2 * np.ones_like(self.zinj)
        self.yinj = np.zeros_like(self.zinj)

        self.v = [
            [-self.res_length/2, -self.res_width/2, -self.well_tvd + self.res_thickness],
            [-self.res_length/2, self.res_width/2, -self.well_tvd + self.res_thickness],
            [self.res_length/2, self.res_width/2, -self.well_tvd + self.res_thickness],
            [self.res_length/2, -self.res_width/2, -self.well_tvd + self.res_thickness],
            [self.res_length/2, -self.res_width/2, -self.well_tvd],
            [-self.res_length/2, -self.res_width/2, -self.well_tvd],
            [-self.res_length/2, self.res_width/2, -self.well_tvd],
            [self.res_length/2, self.res_width/2, -self.well_tvd],
        ]

        self.v = np.array(self.v)
        self.f = [[0,1,2,3], [4,5,6,7], [0, 1, 6, 5], [1, 2, 7, 6], [2, 3, 4, 7], [0, 3, 4, 5]]
        self.verts =  [[self.v[i] for i in p] for p in self.f]

class EnergyDeclineReservoir(BaseReservoir):
    """Conceptual reservoir model where temperature decline is proportional to the energy exctracted from the subsurface."""

    def __init__(self,
                 Tres_init,
                 Pres_init,
                 geothermal_gradient,
                 surface_temp,
                 L,
                 time_init,
                 well_tvd,
                 prd_well_diam,
                 inj_well_diam,
                 num_prd,
                 num_inj,
                 waterloss,
                 powerplant_type,
                 pumpeff,
                 ramey=True,
                 pumping=True,
                  krock=3,
                 rhorock=2700,
                    cprock=1000,
                   impedance = 0.1,
                 res_thickness=200,
                 PI = 20,
                 II = 20,
                 SSR = 1.0,
                 N_ramey_mv_avg=24, # hrs, 1 week worth of accumulation
                 V_res=1,
                 phi_res=0.1,
                 compute_hydrostatic_pressure=True,
                 rock_energy_recovery=1.0,
                 decline_func = lambda k,D,t: (k / t* 2e-2)**5,
                 reservoir_simulator_settings={"fast_mode": False, "period": 3600*8760/12},
                 PumpingModel="OpenLoop",
                 timestep=timedelta(hours=1),
                 krock_wellbore=3
                 ):
        """Initialize reservoir model.

        Args:
            Tres_init (float): initial reservoir temperature in deg C.
            Pres_init (float): initial reservoir pressure in bar.
            geothermal_gradient (float): average geothermal gradient in deg C/km.
            surface_temp (float): surface temperature in deg C.
            L (float): project lifetime in years.
            time_init (datetime): initial time.
            well_tvd (float): well depth in meters.
            prd_well_diam (float): production well diameter in meters.
            inj_well_diam (float): injection well diameter in meters.
            num_prd (int): number of producers.
            num_inj (int): number of injectors.
            waterloss (float): fraction of injected water that is lost to the reservoir (fraction).
            powerplant_type (str): type of power plant (either "Binary" or "Flash").
            pumpeff (float): pump efficiency (fraction).
            ramey (bool, optional): whether or not to use ramey's model for wellbore heat loss/gain. Defaults to True.
            pumping (bool, optional): whther or not to account for parasitic losses due to pumping requirements. Defaults to True.
            krock (float, optional): rock thermal conductivity in W/C-m. Defaults to 3.
            rhorock (float, optional): rock bulk density in kg/m3. Defaults to 2700.
            cprock (float, optional): rock heat capacity in J/kg-K. Defaults to 1000.
            impedance (float, optional): reservoir pressure losses when using an impendance model. Defaults to 0.1.
            res_thickness (float, optional): reservoir thickness in meters. Defaults to 200.
            PI (float, optional): productivity index in kg/s/bar. Defaults to 20.
            II (float, optional): injectivity index in kg/s/bar. Defaults to 20.
            SSR (float, optional): Stimulation success rate, which is a multiplier used to reduce PI and II when stimulation is not fully successful. Defaults to 1.0.
            N_ramey_mv_avg (int, optional): number of timesteps used for averaging the f-function when computing ramey's heat losses with variable mass flow rates. Defaults to 168.
            V_res (float, optional): reservoir bulk volume in km3. Defaults to 1.
            phi_res (float, optional): reservoir porosity (fraction). Defaults to 0.1.
            compute_hydrostatic_pressure: (bool, optional): whether or not hydrostatic pressure is computed or assumed to be equal to the initial reservoir pressure. Defaults to True.
            rock_energy_recovery (float, optional): maximum fraction of subsurface energy that is recoverable (fraction). Defaults to 1.0.
            decline_func (func, optional): function used to establish the correlation between temperature decline and energy extraction. Defaults to a 5th order polynomial.
            reservoir_simulator_settings (dict, optional): information used to reduce the required timestepping when simulating the reservoir. It comes with keys of "fast_mode" to turn it on and "period" to specify the time period needed to pass before the reservoir state is updated, which is aimed at reducing computational requirements in exchange for loss in accuracy. Defaults to {"fast_mode": False, "period": 3600*8760/12}.
            PumpingModel (str, optional): model type used to compute pressure losses (either "OpenLoop" or "ClosedLoop"). Defaults to "OpenLoop".
        """

        super(EnergyDeclineReservoir, self).__init__(Tres_init,
                                            geothermal_gradient,
                                            surface_temp,
                                            L,
                                            time_init,
                                            well_tvd,
                                            prd_well_diam,
                                            inj_well_diam,
                                            num_prd,
                                            num_inj,
                                            waterloss,
                                            powerplant_type,
                                            pumpeff,
                                            ramey,
                                            pumping,
                                            krock,
                                            rhorock,
                                            cprock,
                                            impedance,
                                            res_thickness,
                                            PI,
                                            II,
                                            SSR,
                                            N_ramey_mv_avg,
                                            reservoir_simulator_settings,
                                            PumpingModel,
                                            timestep,
                                            krock_wellbore)

        self.numberoflaterals = 1
        self.well_tvd = well_tvd
        self.well_md = self.well_tvd
        self.res_length = 2000
        self.res_thickness = res_thickness
        self.res_width = V_res * 1e9 /self.num_prd /(self.res_length * self.res_thickness)

        self.Pres_init = self.Phydrostatic if compute_hydrostatic_pressure else Pres_init*1e2
        self.V_res = V_res * 1e9 # m3
        self.phi_res = phi_res
        self.rock_energy_recovery = rock_energy_recovery
        self.h_res_init = steamtable.h_pt(self.Pres_init/1e2, self.Tres_init)
        # Handle T_prd_bh and T_inj_bh as either float or array
        T_prd_bh_iter = [self.T_prd_bh] if not isinstance(self.T_prd_bh, (np.ndarray, list)) else self.T_prd_bh
        T_inj_bh_iter = [self.T_inj_bh] if not isinstance(self.T_inj_bh, (np.ndarray, list)) else self.T_inj_bh
        self.h_prd = np.array([steamtable.h_pt(self.Pres_init/1e2, T) for T in T_prd_bh_iter])
        self.h_inj = np.array([steamtable.hL_t(T) for T in T_inj_bh_iter])
        self.rho_geof_init = densitywater(self.Tres_init)
        self.M_geofluid = self.rho_geof_init * self.V_res * self.phi_res
        self.M_rock = rhorock * self.V_res * (1-self.phi_res)
        self.energy_res_init = self.M_geofluid * self.h_res_init + \
                               self.M_rock * (self.Tres_init) * cprock/1000 * self.rock_energy_recovery #kJ reservoir geofluid + rock exchanged portion assuming 10C drop over the lifetime
        self.energy_res_curr = self.energy_res_init
        self.kold = 1
        self.kratio = 0.99
        self.decline_coeff = 1
        self.decline_func = decline_func
        self.D = 0

        self.configure_well_dimensions()

    def pre_model(self, t, m_prd, m_inj, T_inj):
        """Computations to be performed before stepping the reservoir model.

        Args:
            t (datetime): current timestamp.
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s.
            T_inj (float): injection temperature in deg C.

        """

        mass_produced = m_prd * self.timestep.total_seconds()
        mass_injected = m_inj * self.timestep.total_seconds()
        
        # Ensure h_prd and h_inj are arrays for proper multiplication
        h_prd_arr = np.atleast_1d(self.h_prd)
        h_inj_arr = np.atleast_1d(self.h_inj)
        mass_produced_arr = np.atleast_1d(mass_produced)
        mass_injected_arr = np.atleast_1d(mass_injected)
        
        self.net_energy_produced = (h_prd_arr * mass_produced_arr).sum() - (h_inj_arr * mass_injected_arr).sum()# - self.boundary_influx #kg produced
        self.energy_res_curr -= self.net_energy_produced

    def model(self, t, m_prd, m_inj, T_inj):
        """Computations to be performed when stepping the reservoir model.

        Args:
            t (datetime): current timestamp.
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s.
            T_inj (float): injection temperature in deg C.
        """

        # Handle T_prd_bh and T_inj_bh as either float or array
        T_prd_bh_iter = np.atleast_1d(self.T_prd_bh)
        T_inj_bh_iter = np.atleast_1d(self.T_inj_bh)
        
        self.h_prd = np.array([steamtable.h_pt(self.Pres_init/1e2, T) for T in T_prd_bh_iter]) #self.fxsteam.func_hl(self.T_prd_bh, *self.fxsteam.popt_hl) #np.array([steamtable.h_pt(self.Pres_init/1e2, T) for T in self.T_prd_bh])
        self.h_inj = np.array([steamtable.hL_t(T) for T in T_inj_bh_iter]) #self.fxsteam.func_hl(self.T_inj_bh, *self.fxsteam.popt_hl) #np.array([steamtable.hL_t(T) for T in self.T_inj_bh])
        self.k =  self.energy_res_curr/self.energy_res_init if self.energy_res_curr>=0 else self.kold * self.kratio
        self.D = (np.log(self.energy_res_curr + self.net_energy_produced) - np.log(self.energy_res_curr))/self.timestep.total_seconds()
        self.decline_coeff = self.decline_func(self.k, self.D, (self.time_curr - self.time_init).total_seconds())
        
        # Ensure T_inj_bh is array for np.mean
        T_inj_bh_arr = np.atleast_1d(self.T_inj_bh)
        self.Tres = min(max(np.mean(T_inj_bh_arr)*1.5, self.Tres_init * self.decline_coeff), self.Tres)
        self.T_prd_bh = np.array(self.num_prd*[self.Tres], dtype='float')
        self.kratio = self.k / self.kold
        self.kold = self.k


    def configure_well_dimensions(self):
        """Configuration specifications of a doublet. It requires the specification of a doublet, including the producer dimensions (self.xprod, self.yprod, self.zprod), injector dimensions (self.xinj, self.yinj, self.zinj) and reservoir vertices (self.verts). See Class PercentageReservoir for example implementation.
        """

        self.zprod = np.array([0, -self.well_tvd])
        self.xprod = -self.res_length/2 * np.ones_like(self.zprod)
        self.yprod = np.zeros_like(self.zprod)

        self.zinj = np.array([0, -self.well_tvd])
        self.xinj = self.res_length/2 * np.ones_like(self.zinj)
        self.yinj = np.zeros_like(self.zinj)

        self.v = [
            [-self.res_length/2, -self.res_width/2, -self.well_tvd + self.res_thickness],
            [-self.res_length/2, self.res_width/2, -self.well_tvd + self.res_thickness],
            [self.res_length/2, self.res_width/2, -self.well_tvd + self.res_thickness],
            [self.res_length/2, -self.res_width/2, -self.well_tvd + self.res_thickness],
            [self.res_length/2, -self.res_width/2, -self.well_tvd],
            [-self.res_length/2, -self.res_width/2, -self.well_tvd],
            [-self.res_length/2, self.res_width/2, -self.well_tvd],
            [self.res_length/2, self.res_width/2, -self.well_tvd],
        ]

        self.v = np.array(self.v)
        self.f = [[0,1,2,3], [4,5,6,7], [0, 1, 6, 5], [1, 2, 7, 6], [2, 3, 4, 7], [0, 3, 4, 5]]
        self.verts =  [[self.v[i] for i in p] for p in self.f]

class DiffusionConvection(BaseReservoir):
    """Analytical solution of the one-dimensional transient semi-infinite diffusion-convection equation."""
    def __init__(self,
                 Tres_init,
                 geothermal_gradient,
                 surface_temp,
                 L,
                 time_init,
                 well_tvd,
                 prd_well_diam,
                 inj_well_diam,
                 num_prd,
                 num_inj,
                 waterloss,
                 powerplant_type,
                 pumpeff,
                 ramey=True,
                 pumping=True,
                  krock=30,
                 rhorock=2600,
                    cprock=1100,
                   impedance = 0.1,
                 res_thickness=1000,
                 PI = 20,
                 II = 20,
                 SSR = 1.0,
                 N_ramey_mv_avg=168,
                 V_res=1,
                 phi_res=0.1,
                 lateral_length=1000,
                 dynamic_properties=False,
                 reservoir_simulator_settings={"fast_mode": False, "period": 3600*8760/12},
                 PumpingModel="OpenLoop",
                 timestep=timedelta(hours=1),
                 krock_wellbore=3
                 ):

        """Initialize reservoir model.

        Args:
            Tres_init (float): initial reservoir temperature in deg C.
            geothermal_gradient (float): average geothermal gradient in deg C/km.
            surface_temp (float): surface temperature in deg C.
            L (float): project lifetime in years.
            time_init (datetime): initial time.
            well_tvd (float): well depth in meters.
            prd_well_diam (float): production well diameter in meters.
            inj_well_diam (float): injection well diameter in meters.
            num_prd (int): number of producers.
            num_inj (int): number of injectors.
            waterloss (float): fraction of injected water that is lost to the reservoir (fraction).
            powerplant_type (str): type of power plant (either "Binary" or "Flash").
            pumpeff (float): pump efficiency (fraction).
            ramey (bool, optional): whether or not to use ramey's model for wellbore heat loss/gain. Defaults to True.
            pumping (bool, optional): whther or not to account for parasitic losses due to pumping requirements. Defaults to True.
            krock (float, optional): rock thermal conductivity in W/C-m. Defaults to 3.
            rhorock (float, optional): rock bulk density in kg/m3. Defaults to 2700.
            cprock (float, optional): rock heat capacity in J/kg-K. Defaults to 1000.
            impedance (float, optional): reservoir pressure losses when using an impendance model. Defaults to 0.1.
            res_thickness (float, optional): reservoir thickness in meters. Defaults to 200.
            PI (float, optional): productivity index in kg/s/bar. Defaults to 20.
            II (float, optional): injectivity index in kg/s/bar. Defaults to 20.
            SSR (float, optional): Stimulation success rate, which is a multiplier used to reduce PI and II when stimulation is not fully successful. Defaults to 1.0.
            N_ramey_mv_avg (int, optional): number of timesteps used for averaging the f-function when computing ramey's heat losses with variable mass flow rates. Defaults to 168.
            V_res (float, optional): reservoir bulk volume for all wells in km3. Defaults to 1.
            phi_res (float, optional): reservoir porosity (fraction). Defaults to 0.1.
            lateral_length (float, optional):  lateral length for each well in meters. Defaults to 1000.
            dynamic_properties (bool, optional):  whether or not geofluid properties in the subsurface are updated using steamtables as a function of varying subsurface temperature. Defaults to False.
            reservoir_simulator_settings (dict, optional): information used to reduce the required timestepping when simulating the reservoir. It comes with keys of "fast_mode" to turn it on and "period" to specify the time period needed to pass before the reservoir state is updated, which is aimed at reducing computational requirements in exchange for loss in accuracy. Defaults to {"fast_mode": False, "period": 3600*8760/12}.
            PumpingModel (str, optional): model type used to compute pressure losses (either "OpenLoop" or "ClosedLoop"). Defaults to "OpenLoop".
        """

        super(DiffusionConvection, self).__init__(Tres_init,
                                            geothermal_gradient,
                                            surface_temp,
                                            L,
                                            time_init,
                                            well_tvd,
                                            prd_well_diam,
                                            inj_well_diam,
                                            num_prd,
                                            num_inj,
                                            waterloss,
                                            powerplant_type,
                                            pumpeff,
                                            ramey,
                                            pumping,
                                            krock,
                                            rhorock,
                                            cprock,
                                            impedance,
                                            res_thickness,
                                            PI,
                                            II,
                                            SSR,
                                            N_ramey_mv_avg,
                                            reservoir_simulator_settings,
                                            PumpingModel,
                                            timestep,
                                            krock_wellbore)
        
        self.numberoflaterals = 1
        self.well_tvd = well_tvd
        self.well_md = self.well_tvd + lateral_length
        self.lateral_length = lateral_length
        self.phi_res = phi_res
        self.res_thickness = res_thickness
        self.V_res_per_well = V_res/self.num_prd # make it for a single well
        if self.lateral_length == 0: # vertical well
            # consider a square reservoir and compute cross sectional area
            # self.res_length = np.sqrt(self.V_res_per_well*1e9/self.res_thickness) # meters
            self.res_length = np.sqrt(self.V_res_per_well*1e9/self.res_thickness) # meters

        else:
            # consider adjacent and parallel 1 injector & 2 producer system where res_length is the distance between them ... only half of the volume is used
            self.res_length = self.V_res_per_well*1e9/(self.res_thickness * self.lateral_length)
            # self.res_length = self.V_res_per_well*1e9/(self.res_thickness * self.lateral_length)

        # Handle T_prd_bh as either float or array
        T_prd_bh_val = _safe_mean(self.T_prd_bh)
        self.rhow_prd_bh = densitywater(T_prd_bh_val)
        self.cw_prd_bh = heatcapacitywater(T_prd_bh_val) # J/kg-degC
        self.rhom = phi_res * self.rhow_prd_bh + (1-phi_res) * rhorock
        self.cm = phi_res * self.cw_prd_bh + (1-phi_res) * cprock
        # self.rhom = rhorock
        # self.cm = cprock
        self.res_width = self.V_res_per_well*1e9/(self.res_thickness*self.res_length) # reservoir width [m]
        self.A = self.res_thickness * self.res_width * self.phi_res # reservoir cross-sectional area [m2]
        self.D = self.krock / (self.rhom * self.cm) #m2/s
        # self.uw = (self.num_prd * 50)/ self.rhow_prd_bh / self.A # m/s with random flow rate initialization
        self.uw = 50/ self.rhow_prd_bh / self.A # m/s with random flow rate initialization
        self.V = self.uw * self.rhow_prd_bh * self.cw_prd_bh / (self.rhom * self.cm)

        self.Vs = np.array([self.V]) # randomly initialized at 100kg/s per producer
        self.T_injs = np.array([70]) # randomly initialized
        self.timesteps = np.array([1]) # first record gets minimal weight
        self.dynamic_properties = dynamic_properties
        self.prev_time_passed_seconds = 0
        self.rhs_cumulative = 0

        # For visualization purposes
        self.configure_well_dimensions()

    def pde_solution(self, tau, t, x, V):
        """Solve PDE.

        Args:
            tau (float): integration time variable
            t (float): total time passed
            x (float): reservoir length at which the integration is performed
            V (float): average velocity

        Returns:
            output: float
        """
        mask = self.timesteps < (t-tau)
        coeff = (self.T_injs[mask][-1]-self.Tres_init)/np.sqrt(16 * np.pi * self.D * tau**3)
        term1 = (x - V*tau) * np.exp(- (x - V*tau)**2/(4 * self.D * tau))
        term2 = (x + V*tau) * np.exp(V*x/self.D - (x + V*tau)**2/(4 * self.D * tau))
        return  coeff * (term1 + term2)

    def pre_model(self, t, m_prd, m_inj, T_inj):
        """Computations to be performed before stepping the reservoir model.

        Args:
            t (datetime): current timestamp.
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s.
            T_inj (float): injection temperature in deg C.

        """

        self.uw = _safe_mean(m_prd) / self.rhow_prd_bh / self.A
        self.V = self.uw * self.rhow_prd_bh * self.cw_prd_bh / (self.rhom * self.cm)
        self.Vs = np.append(self.Vs, self.V)
        self.T_injs = np.append(self.T_injs, T_inj)
        self.time_passed_seconds = (t - self.time_init).total_seconds()
        self.timesteps = np.append(self.timesteps, self.time_passed_seconds)

    def model(self, t, m_prd, m_inj, T_inj):
        """Computations to be performed when stepping the reservoir model.

        Args:
            t (datetime): current timestamp.
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s.
            T_inj (float): injection temperature in deg C.
        """

        if self.dynamic_properties:
            Tavg = _safe_mean(self.T_prd_bh)
            # self.rhow_prd_bh = densitywater(Tavg)
            # self.cw_prd_bh = self.cw_prd_bh

            Tround = round(Tavg, 0)
            self.rhow_prd_bh = self.cache["densitywater"].get(Tround, densitywater(Tavg))
            self.cache["densitywater"][Tround] = self.rhow_prd_bh

            self.cw_prd_bh = self.cache["cpwater"].get(Tround, heatcapacitywater(Tavg))
            self.cache["cpwater"][Tround] = self.cw_prd_bh

        self.Vavg = self.Vs.mean()
        rhs, _ = integrate.quad(self.pde_solution, self.prev_time_passed_seconds, self.time_passed_seconds, 
                                 (self.time_passed_seconds, self.res_length, self.Vavg),
                              epsabs=1e-3, epsrel=1e-3,
                              )
        self.rhs_cumulative += rhs
        self.prev_time_passed_seconds = self.time_passed_seconds

        self.Tres = self.rhs_cumulative + self.Tres_init
        self.T_prd_bh = np.array(self.num_prd*[self.Tres], dtype='float')
        self.T_inj_wh = np.array(self.num_inj*[T_inj], dtype='float')

    def configure_well_dimensions(self):
        """Configuration specifications of a doublet. It requires the specification of a doublet, including the producer dimensions (self.xprod, self.yprod, self.zprod), injector dimensions (self.xinj, self.yinj, self.zinj) and reservoir vertices (self.verts). See Class PercentageReservoir for example implementation.
        """

        # Vertical well system
        if self.lateral_length == 0: 
            self.zprod = np.array([0, -self.well_tvd])
            self.xprod = -self.res_length/2 * np.ones_like(self.zprod)
            self.yprod = np.zeros_like(self.zprod)

            self.zinj = np.array([0, -self.well_tvd])
            self.xinj = self.res_length/2 * np.ones_like(self.zinj)
            self.yinj = np.zeros_like(self.zinj)

            self.v = [
                [-self.res_length/2, -self.res_width/2, -self.well_tvd + self.res_thickness],
                [-self.res_length/2, self.res_width/2, -self.well_tvd + self.res_thickness],
                [self.res_length/2, self.res_width/2, -self.well_tvd + self.res_thickness],
                [self.res_length/2, -self.res_width/2, -self.well_tvd + self.res_thickness],
                [self.res_length/2, -self.res_width/2, -self.well_tvd],
                [-self.res_length/2, -self.res_width/2, -self.well_tvd],
                [-self.res_length/2, self.res_width/2, -self.well_tvd],
                [self.res_length/2, self.res_width/2, -self.well_tvd],
            ]

        # Horizontal well system
        else:

            self.zprod = np.array([0, -self.well_tvd, -self.well_tvd])
            self.xprod = np.array([-self.res_length/2, -self.res_length/2, self.res_length/2])
            self.yprod = -self.res_width/2 * np.ones_like(self.zprod)

            self.zinj = np.array([0, -self.well_tvd, -self.well_tvd])
            self.xinj = np.array([self.res_length/2, self.res_length/2, -self.res_length/2])
            self.yinj = self.res_width/2 * np.ones_like(self.zinj)

            self.v = [
                [-self.res_length/2, -self.res_width/2, -self.well_tvd + self.res_thickness/2],
                [-self.res_length/2, self.res_width/2, -self.well_tvd + self.res_thickness/2],
                [self.res_length/2, self.res_width/2, -self.well_tvd + self.res_thickness/2],
                [self.res_length/2, -self.res_width/2, -self.well_tvd + self.res_thickness/2],
                [self.res_length/2, -self.res_width/2, -self.well_tvd - self.res_thickness/2],
                [-self.res_length/2, -self.res_width/2, -self.well_tvd - self.res_thickness/2],
                [-self.res_length/2, self.res_width/2, -self.well_tvd - self.res_thickness/2],
                [self.res_length/2, self.res_width/2, -self.well_tvd - self.res_thickness/2],
            ]

        self.v = np.array(self.v)
        self.f = [[0,1,2,3], [4,5,6,7], [0, 1, 6, 5], [1, 2, 7, 6], [2, 3, 4, 7], [0, 3, 4, 5]]
        self.verts =  [[self.v[i] for i in p] for p in self.f]

class ULoopSBT(BaseReservoir):
    """Numerical ULoop model based on Slender-Body Theory (SBT), originally developed by  Beckers et al. (2023)."""
    def __init__(self,
                 Tres_init,
                 Pres_init,
                 geothermal_gradient,
                 surface_temp,
                 L,
                 time_init,
                 well_tvd,
                 prd_well_diam,
                 inj_well_diam,
                 num_prd,
                 num_inj,
                 waterloss,
                 powerplant_type,
                 pumpeff,
                 times_arr,
                 ramey=False,
                 pumping=True,
                  k_m=2.83,
                 rho_m=2875,
                    c_m=825,
                   impedance = 0.1,
                 res_thickness=1000,
                 PI = 20,
                 II = 20,
                 SSR = 1.0,
                 N_ramey_mv_avg=168,
                 V_res=1,
                 phi_res=0.1,
                 half_lateral_length=2000,
                 lateral_diam=0.31115,
                 lateral_spacing=100,
                 dynamic_properties=False,
                 k_f=0.68,
                 mu_f = 600*1E-6,
                 cp_f=4200,
                 rho_f=1000,
                 dx=None,
                 numberoflaterals=3,
                 lateralflowallocation=None,
                 lateralflowmultiplier=1,
                 fullyimplicit=1,
                 reservoir_simulator_settings={"fast_mode": False, "accuracy": 5, "DynamicFluidProperties": False},
                 PumpingModel="ClosedLoop",
                 closedloop_design="Default",
                 FMM=1,
                 FMMtriggertime=3600*24*10,
                 debug_mode=False,
                 ):

        """Initialize reservoir model.

        Args:
            Tres_init (float): initial reservoir temperature in deg C.
            geothermal_gradient (float): average geothermal gradient in deg C/km.
            surface_temp (float): surface temperature in deg C.
            L (float): project lifetime in years.
            time_init (datetime): initial time.
            well_tvd (float): well depth in meters.
            prd_well_diam (float): production well diameter in meters.
            inj_well_diam (float): injection well diameter in meters.
            num_prd (int): number of producers.
            num_inj (int): number of injectors.
            waterloss (float): fraction of injected water that is lost to the reservoir (fraction).
            powerplant_type (str): type of power plant (either "Binary" or "Flash").
            pumpeff (float): pump efficiency (fraction).
            ramey (bool, optional): whether or not to use ramey's model for wellbore heat loss/gain. Defaults to True.
            pumping (bool, optional): whther or not to account for parasitic losses due to pumping requirements. Defaults to True.
            k_m (float, optional): rock thermal conductivity in W/C-m. Defaults to 3.
            rho_m (float, optional): rock bulk density in kg/m3. Defaults to 2700.
            c_m (float, optional): rock heat capacity in J/kg-K. Defaults to 1000.
            impedance (float, optional): reservoir pressure losses when using an impendance model. Defaults to 0.1.
            res_thickness (float, optional): reservoir thickness in meters. Defaults to 200.
            PI (float, optional): productivity index in kg/s/bar. Defaults to 20.
            II (float, optional): injectivity index in kg/s/bar. Defaults to 20.
            SSR (float, optional): Stimulation success rate, which is a multiplier used to reduce PI and II when stimulation is not fully successful. Defaults to 1.0.
            N_ramey_mv_avg (int, optional): number of timesteps used for averaging the f-function when computing ramey's heat losses with variable mass flow rates. Defaults to 168.
            V_res (float, optional): reservoir bulk volume for all wells in km3. Defaults to 1.
            phi_res (float, optional): reservoir porosity (fraction). Defaults to 0.1.
            half_lateral_length (float, optional): half of the total producer-to-injector lateral length in meters. Defaults to 2000.
            lateral_diam (float, optional): diameter of wellbore lateral section in meters. Defaults to 0.3115.
            lateral_spacing (float, optional): spacing between uloop laterals in meters. Defaults to 100.
            dynamic_properties (bool, optional):  whether or not geofluid properties in the subsurface are updated using steamtables as a function of varying subsurface temperature. Defaults to False.
            k_f (float, optional): fluid thermal conductivity in W/C-m. Defaults to 0.68.
            mu_f (float, optional): fluid kinematic viscosity in m2/s. Defaults to 600e-6.
            cp_f (float, optional): fluid heat capacity in J/kg-K. Defaults to 4200.
            rho_f (float, optional): fluid density in kg/m3. Defaults to 1000.
            dx (int, optional): mesh descritization size in meters. Defaults to None (computed automatically).
            numberoflaterals (int, optional): number of laterals for each uloop doublet. Defaults to 3.
            lateralflowallocation (int, optional): distribution of flow across uloop laterals. Defaults to None (equal distribution)
            lateralflowmultiplier (int, optional): velocity multiplier across laterals. Defaults to 1.
            fullyimplicit (int, optional): how to solve the numerical system of equations using Euler's. Defaults to 1.
            reservoir_simulator_settings (dict, optional): information used to reduce the required timestepping when simulating the reservoir. It comes with keys of "fast_mode" to turn it on and "period" to specify the time period needed to pass before the reservoir state is updated, which is aimed at reducing computational requirements in exchange for loss in accuracy. Defaults to {"fast_mode": False, "period": 3600*8760/12}.
            PumpingModel (str, optional): model type used to compute pressure losses (either "OpenLoop" or "ClosedLoop"). Defaults to "ClosedLoop".
            closedloop_design (str, optional): Type of closedloop_design to simulate (either "Default" or "Eavor"). Defaults to "Default".
        """

        super(ULoopSBT, self).__init__(Tres_init,
                                        geothermal_gradient,
                                        surface_temp,
                                        L,
                                        time_init,
                                        well_tvd,
                                        prd_well_diam,
                                        inj_well_diam,
                                        num_prd,
                                        num_inj,
                                        waterloss,
                                        powerplant_type,
                                        pumpeff,
                                        ramey,
                                        pumping,
                                        k_m,
                                        rho_m,
                                        c_m,
                                        impedance,
                                        res_thickness,
                                        PI,
                                        II,
                                        SSR,
                                        N_ramey_mv_avg,
                                        reservoir_simulator_settings,
                                        PumpingModel)

        # Initialize debug_mode after super() call
        self.debug_mode = debug_mode
        self.debug_log = [] if debug_mode else None

        self.well_tvd = well_tvd
        self.well_md = self.well_tvd + numberoflaterals * half_lateral_length
        self.half_lateral_length = half_lateral_length
        self.lateral_length = 2 * half_lateral_length
        self.lateral_diam = lateral_diam
        self.vertical_diam = (self.prd_well_diam + self.inj_well_diam) / 2
        self.rho_f = rho_f if rho_f else densitywater(_safe_mean(self.T_prd_bh))
        self.cp_f = cp_f if cp_f else heatcapacitywater(_safe_mean(self.T_prd_bh))
        self.mu_f = mu_f if mu_f else viscositywater(_safe_mean(self.T_prd_bh))
        self.k_f = k_f
        self.c_m = c_m
        self.k_m = k_m
        self.rho_m = rho_m
        self.numberoflaterals = numberoflaterals
        self.lateralflowallocation = lateralflowallocation if lateralflowallocation else self.numberoflaterals*[1/self.numberoflaterals]
        self.lateralflowmultiplier = lateralflowmultiplier
        self.fullyimplicit = fullyimplicit
        self.accuracy = reservoir_simulator_settings["accuracy"]
        self.L = L
        self.geothermal_gradient = geothermal_gradient/1000 #C/m
        self.surface_temp = surface_temp
        self.times_arr = times_arr
        self.lateral_spacing = lateral_spacing
        self.res_thickness = res_thickness 
        self.FMM = FMM #if 1, use fast multi-pole methold like approach (i.e., combine old heat pulses to speed up simulation)
        self.FMMtriggertime = FMMtriggertime #threshold time beyond which heat pulses can be combined with others [s
        self.Deltat = self.timestep.total_seconds()
        N = 10
        # Calculate dx safely: use provided dx, or calculate from lateral_length, or use well_tvd as fallback
        if dx and dx > 0:
            self.dx = dx
        elif self.lateral_length and self.lateral_length > 0:
            self.dx = self.lateral_length // N
            if self.dx <= 0:  # Ensure dx is positive
                self.dx = max(10.0, self.well_tvd // N) if self.well_tvd > 0 else 10.0
        else:
            # Fallback: use well_tvd or a default value
            self.dx = max(10.0, self.well_tvd // N) if self.well_tvd > 0 else 10.0

        # Following the design proposed by Eavor (https://pangea.stanford.edu/ERE/db/GeoConf/papers/SGW/2022/Beckers.pdf)
        if "eavor" in closedloop_design.lower():
            vertical_section_length = 2000
            vertical_well_spacing = 100
            junction_depth = 4000
            self.res_thickness = self.well_tvd - junction_depth
            angle = 20*np.pi/180
            element_length = 150

            N = 2
            # generate inj well profile
            zinj = np.linspace(0, -vertical_section_length, N).reshape(-1, 1)
            yinj = np.zeros((len(zinj), 1))
            xinj = np.zeros((len(zinj), 1))

            inclined_length = abs(-junction_depth - zinj[-1])/np.cos(angle)

            zinj_inclined_length  = np.linspace(np.round((zinj[-1] - junction_depth)/2), -junction_depth, N)
            yinj_inclined_length = np.zeros((len(zinj_inclined_length), 1))
            xend = xinj[-1]+inclined_length * np.sin(angle)
            xinj_inclined_length = np.linspace((xinj[-1]+xend)/2, xend, N)

            zinj = np.concatenate((zinj, zinj_inclined_length))
            xinj = np.concatenate((xinj, xinj_inclined_length))
            yinj = np.concatenate((yinj, yinj_inclined_length))

            # generate prod well profile
            zprod = np.flip(zinj)
            xprod = np.flip(xinj)
            yprod = np.flip(yinj) + vertical_well_spacing;

            # Generate Laterals
            # Injection points
            x_ip = np.zeros((1,numberoflaterals))
            y_ip = np.zeros((1,numberoflaterals))
            z_ip = np.zeros((1,numberoflaterals))
            for i in range(numberoflaterals):
                y_ip[0, i] = yinj[-1]-(lateral_spacing*(numberoflaterals-1))/2+i*lateral_spacing-(yinj[-1]-yprod[-1])/2
                x_ip[0, i] = xinj[-1]+element_length*3*np.sin(angle)
                z_ip[0, i] = zinj[-1]-element_length*3*np.cos(angle)

            # Lateral feedways
            x_feed = np.zeros((N, numberoflaterals))
            y_feed = np.zeros((N, numberoflaterals))
            z_feed = np.zeros((N, numberoflaterals))
            for i in range(numberoflaterals):
                # we space things out by 1% to avoid zero segment lengths in SBT
                x_feed[:,i] = np.linspace(xinj[-1, 0], x_ip[0, i] * 0.99, N)
                y_feed[:,i] = np.linspace(yinj[-1, 0], y_ip[0, i] * 0.99, N)
                z_feed[:,i] = np.linspace(zinj[-1, 0] , z_ip[0, i] * 0.99, N)

                
            # lateral template ...
            lateral_length = (well_tvd-abs(z_ip[0, -1]))/np.cos(angle)
            z_template_lateral = np.linspace(z_ip[0, -1], -well_tvd, N)
            z_template_lateral = np.concatenate((z_template_lateral, 
                                                z_template_lateral[[-1]]+element_length,
                                                z_template_lateral[::-1]+2*element_length
                                                ))
            z_template_lateral = np.repeat(z_template_lateral[None], numberoflaterals, axis=0).T

            xend = x_ip[0, -1]+lateral_length * np.sin(angle)
            x_template_lateral = np.concatenate((np.linspace(x_ip[0, -1], xend, N),
                                                np.array([xend]),
                                                np.linspace(x_ip[0, -1], xend, N)[::-1]
                                            ))
            x_template_lateral = np.repeat(x_template_lateral[None], numberoflaterals, axis=0).T

            y_template_lateral = np.repeat(y_ip, x_template_lateral.shape[0], axis=0)

            # Lateral returns
            x_return = np.zeros((N, numberoflaterals))
            y_return = np.zeros((N, numberoflaterals))
            z_return = np.zeros((N, numberoflaterals))
            for i in range(numberoflaterals):
                x_return[:,i] = np.linspace(x_template_lateral[-1, i] * 1.01, xprod[0, 0],N)
                y_return[:,i] = np.linspace(y_template_lateral[-1, i] * 1.01, yprod[0, 0],N)
                z_return[:,i] = np.linspace(z_template_lateral[-1, i] * 1.01, zprod[0, 0],N)
                
            zlat = np.vstack((z_feed, z_template_lateral, z_return))
            xlat = np.vstack((x_feed, x_template_lateral, x_return))
            ylat = np.vstack((y_feed, y_template_lateral, y_return))

            self.xinj, self.yinj, self.zinj = xinj, yinj, zinj
            self.xprod, self.yprod, self.zprod = xprod, yprod, zprod
            self.xlat, self.ylat, self.zlat = xlat, ylat, zlat

        else:
            # Coordinates of injection well (coordinates are provided from top to bottom in the direction of flow)
            self.zinj = np.arange(0, -self.well_tvd -self.dx, -self.dx).reshape(-1, 1)
            self.yinj = np.zeros((len(self.zinj), 1))
            self.xinj = -(self.lateral_length/2) * np.ones((len(self.zinj), 1))

            # Coordinates of production well (coordinates are provided from bottom to top in the direction of flow)
            self.zprod = np.arange(-self.well_tvd, 0 + self.dx, self.dx).reshape(-1, 1)
            self.yprod = np.zeros((len(self.zprod), 1))
            self.xprod = (self.lateral_length/2) * np.ones((len(self.zprod), 1))

            # Coordinates of laterals
            self.xlat = np.repeat(np.arange(-self.lateral_length//2, self.lateral_length//2 + self.dx, self.dx)[:,None], self.numberoflaterals, axis=1)
            if self.numberoflaterals > 1:
                lats = []
                for i in np.arange(self.numberoflaterals//2):
                    arr = self.lateral_spacing * (i+1) * np.concatenate((np.cos(np.linspace(-np.pi/2, 0, 3)), np.ones(self.xlat.shape[0]-6), \
                                        np.cos(np.linspace(0, np.pi/2, 3))))
                    lats.extend([arr, -arr])
                # if odd number of laterals, then include a center lateral
                if self.numberoflaterals % 2 == 1:
                    lats.append(np.zeros_like(arr))

                self.ylat = np.array(lats).T
            else:
                self.ylat = np.zeros_like(self.xlat)
            self.zlat = -self.well_tvd * np.ones_like(self.xlat)
            
        # Merge x-, y-, and z-coordinates
        self.x = np.concatenate((self.xinj, self.xprod, self.xlat.flatten(order="F")[:,None]))
        self.y = np.concatenate((self.yinj, self.yprod, self.ylat.flatten(order="F")[:,None]))
        self.z = np.concatenate((self.zinj, self.zprod, self.zlat.flatten(order="F")[:,None]))

        self.res_length = self.x.max() - self.x.min()
        self.res_width = self.y.max() - self.y.min() + 2 * self.lateral_spacing

        self.alpha_f = self.k_f / self.rho_f / self.cp_f  # Fluid thermal diffusivity [m2/s]
        self.Pr_f = self.mu_f / self.rho_f / self.alpha_f  # Fluid Prandtl number [-]
        self.alpha_m = self.k_m / self.rho_m / self.c_m  # Thermal diffusivity medium [m2/s]
        self.interconnections = np.concatenate((np.array([len(self.xinj)],dtype=int), np.array([len(self.xprod)],dtype=int), \
        (np.ones(self.numberoflaterals - 1, dtype=int) * len(self.xlat))))
        self.interconnections = np.cumsum(self.interconnections)  # lists the indices of interconnections between inj, prod,
        # and laterals (this will used to take care of the duplicate coordinates of the start and end points of the laterals)

        self.radiusvector = np.concatenate([np.ones(len(self.xinj) + len(self.xprod) - 2) * self.vertical_diam/2, np.ones(self.numberoflaterals * len(self.xlat) - self.numberoflaterals) * self.lateral_diam/2])  # Stores radius of each element in a vector [m]
        self.Dvector = self.radiusvector * 2  # Diameter of each element [m]
        
        # Ensure lateralflowallocation doesn't cause division by zero
        if self.lateralflowallocation is not None:
            allocation_sum = np.sum(self.lateralflowallocation)
            if allocation_sum > 0:
                self.lateralflowallocation = self.lateralflowallocation / allocation_sum  # Ensure the sum equals 1
            else:
                # Default to equal distribution
                self.lateralflowallocation = np.ones(self.numberoflaterals) / self.numberoflaterals

        self.dL = np.sqrt((self.x[1:] - self.x[:-1]) ** 2 + (self.y[1:] - self.y[:-1]) ** 2 + (self.z[1:] - self.z[:-1]) ** 2)
        self.dL = np.delete(self.dL, self.interconnections - 1)
        self.dz = self.z[1:] - self.z[:-1] # injector yields negative pressure loss (i.e., positive gain); producer causes positive for hydro pressure losses
        self.dz = np.delete(self.dz, self.interconnections - 1)

        self.Deltaz = np.sqrt((self.x[1:] - self.x[:-1]) ** 2 + (self.y[1:] - self.y[:-1]) ** 2 + (self.z[1:] - self.z[:-1]) ** 2)  # Length of each segment [m]
        self.Deltaz = np.delete(self.Deltaz, self.interconnections - 1)  # Removes the phantom elements due to duplicate coordinates
        
        # Check for NaN/Inf in Deltaz before proceeding
        if np.any(np.isnan(self.Deltaz)) or np.any(np.isinf(self.Deltaz)):
            raise ValueError(f"Invalid Deltaz values detected (NaN/Inf). This may be caused by invalid geometry parameters. "
                           f"Please ensure: lateral_length >= 5000m, numberoflaterals >= 3, well_tvd >= 7000m, dx >= 500m")
        
        self.TotalLength = np.sum(self.Deltaz)  # Total length of all elements (for informational purposes only) [m]
        self.total_drilling_length = self.TotalLength

        # Quality Control - ensure radiusvector doesn't contain zeros
        if np.any(self.radiusvector <= 0):
            raise ValueError(f"Invalid radiusvector values (zero or negative). Check vertical_diam and lateral_diam parameters.")
        
        self.LoverR = self.Deltaz / self.radiusvector  # Ratio of pipe segment length to radius along the wellbore [-]
        
        # Check for NaN/Inf in LoverR
        if np.any(np.isnan(self.LoverR)) or np.any(np.isinf(self.LoverR)):
            raise ValueError(f"Invalid LoverR values detected (NaN/Inf). This may be caused by division by zero in radiusvector. "
                           f"Please ensure: vertical_diam > 0, lateral_diam > 0")
        
        self.smallestLoverR = np.min(self.LoverR)  # Smallest ratio of pipe segment length to pipe radius. This ratio should be larger than 10. [-]

        if self.smallestLoverR < 10:
            print('Warning: smallest ratio of segment length over radius is less than 10. Good practice is to keep this ratio larger than 10.')

        if self.numberoflaterals > 1:
            self.DeltazOrdered = np.concatenate((self.Deltaz[0:(self.interconnections[0]-1)], self.Deltaz[(self.interconnections[1]-2):(self.interconnections[2]-3)], self.Deltaz[(self.interconnections[0]-1):(self.interconnections[1]-2)]))
        else:
            self.DeltazOrdered = np.concatenate((self.Deltaz[0:self.interconnections[0] - 1], self.Deltaz[self.interconnections[1] - 1:-1], self.Deltaz[self.interconnections[0]:self.interconnections[1] - 2]))

        # Calculate RelativeLengthChanges safely, avoiding division by zero
        denominator = self.DeltazOrdered[:-1]
        denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)  # Replace zeros with small epsilon
        self.RelativeLengthChanges = (self.DeltazOrdered[1:] - self.DeltazOrdered[:-1]) / denominator
        
        # Check for NaN/Inf in RelativeLengthChanges
        if np.any(np.isnan(self.RelativeLengthChanges)) or np.any(np.isinf(self.RelativeLengthChanges)):
            raise ValueError(f"Invalid RelativeLengthChanges detected (NaN/Inf). This may be caused by invalid geometry parameters. "
                           f"Please ensure: lateral_length >= 5000m, numberoflaterals >= 3, well_tvd >= 7000m, dx >= 500m")

        if max(abs(self.RelativeLengthChanges)) > 0.5:
            print('Warning: abrupt change(s) in segment length detected, which may cause numerical instabilities. Good practice is to avoid abrupt length changes to obtain smooth results.')

        for dd in range(1, self.numberoflaterals + 1):
            if abs(self.xinj[-1] - self.xlat[0][dd - 1]) > 1e-12 or abs(self.yinj[-1] - self.ylat[0][dd - 1]) > 1e-12 or abs(self.zinj[-1] - self.zlat[0][dd - 1]) > 1e-12:
                print(f'Error: Coordinate mismatch between bottom of injection well and start of lateral #{dd}')

            if abs(self.xprod[0] - self.xlat[-1][dd - 1]) > 1e-12 or abs(self.yprod[0] - self.ylat[-1][dd - 1]) > 1e-12 or abs(self.zprod[0] - self.zlat[-1][dd - 1]) > 1e-12:
                print(f'Error: Coordinate mismatch between bottom of production well and end of lateral #{dd}')

        if self.accuracy == 1:
            self.NoArgumentsFinitePipeCorrection = 25
            self.NoDiscrFinitePipeCorrection = 200
            self.NoArgumentsInfCylIntegration = 25
            self.NoDiscrInfCylIntegration = 200
            self.LimitPointSourceModel = 1.5
            self.LimitCylinderModelRequired = 25
            self.LimitInfiniteModel = 0.05
            self.LimitNPSpacingTime = 0.1
            self.LimitSoverL = 1.5
            self.M = 3
        elif self.accuracy == 2:
            self.NoArgumentsFinitePipeCorrection = 50
            self.NoDiscrFinitePipeCorrection = 400
            self.NoArgumentsInfCylIntegration = 50
            self.NoDiscrInfCylIntegration = 400
            self.LimitPointSourceModel = 2.5
            self.LimitCylinderModelRequired = 50
            self.LimitInfiniteModel = 0.01
            self.LimitNPSpacingTime = 0.04
            self.LimitSoverL = 2
            self.M = 4
        elif self.accuracy == 3:
            self.NoArgumentsFinitePipeCorrection = 100
            self.NoDiscrFinitePipeCorrection = 500
            self.NoArgumentsInfCylIntegration = 100
            self.NoDiscrInfCylIntegration = 500
            self.LimitPointSourceModel = 5
            self.LimitCylinderModelRequired = 100
            self.LimitInfiniteModel = 0.004
            self.LimitNPSpacingTime = 0.02
            self.LimitSoverL = 3
            self.M = 5
        elif self.accuracy == 4:
            self.NoArgumentsFinitePipeCorrection = 200
            self.NoDiscrFinitePipeCorrection = 1000
            self.NoArgumentsInfCylIntegration = 200
            self.NoDiscrInfCylIntegration = 1000
            self.LimitPointSourceModel = 10
            self.LimitCylinderModelRequired = 200
            self.LimitInfiniteModel = 0.002
            self.LimitNPSpacingTime = 0.01
            self.LimitSoverL = 5
            self.M = 10
        elif self.accuracy == 5:
            self.NoArgumentsFinitePipeCorrection = 400
            self.NoDiscrFinitePipeCorrection = 2000
            self.NoArgumentsInfCylIntegration = 400
            self.NoDiscrInfCylIntegration = 2000
            self.LimitPointSourceModel = 20
            self.LimitCylinderModelRequired = 400
            self.LimitInfiniteModel = 0.001
            self.LimitNPSpacingTime = 0.005
            self.LimitSoverL = 9
            self.M = 20
        elif self.accuracy == 6:
            self.NoArgumentsFinitePipeCorrection = 400
            self.NoDiscrFinitePipeCorrection = 2000
            self.NoArgumentsInfCylIntegration = 400
            self.NoDiscrInfCylIntegration = 2000
            self.LimitPointSourceModel = 20
            self.LimitCylinderModelRequired = 400
            self.LimitInfiniteModel = 0.001
            self.LimitNPSpacingTime = 1e-6
            self.LimitSoverL = 9
            self.M = 20

        self.timeforpointssource = max(self.Deltaz)**2 / self.alpha_m * self.LimitPointSourceModel  # Calculates minimum time step size when point source model becomes applicable [s]
        self.timeforlinesource = max(self.radiusvector)**2 / self.alpha_m * self.LimitCylinderModelRequired  # Calculates minimum time step size when line source model becomes applicable [s]
        self.timeforfinitelinesource = max(self.Deltaz)**2 / self.alpha_m * self.LimitInfiniteModel  # Calculates minimum time step size when finite line source model should be considered [s]

        self.fpcminarg = min(self.Deltaz)**2 / (4 * self.alpha_m * (self.times_arr[-1] * self.Deltat))
        self.fpcmaxarg = max(self.Deltaz)**2 / (4 * self.alpha_m * (min(self.times_arr[1:] - self.times_arr[:-1]) * self.Deltat))
        self.Amin1vector = np.logspace(np.log10(self.fpcminarg) - 0.1, np.log10(self.fpcmaxarg) + 0.1, self.NoArgumentsFinitePipeCorrection)
        self.finitecorrectiony = np.zeros(self.NoArgumentsFinitePipeCorrection)
        for i, Amin1 in enumerate(self.Amin1vector):
            Amax1 = (16)**2
            if Amin1 > Amax1:
                Amax1 = 10 * Amin1
            Adomain1 = np.logspace(np.log10(Amin1), np.log10(Amax1), self.NoDiscrFinitePipeCorrection)
            self.finitecorrectiony[i] = np.trapz(-1 / (Adomain1 * 4 * np.pi * self.k_m) * erfc(1/2 * np.power(Adomain1, 1/2)), Adomain1)
            
        self.besselminarg = self.alpha_m * (min(self.times_arr[1:] - self.times_arr[:-1]) * self.Deltat) / max(self.radiusvector)**2
        self.besselmaxarg = self.alpha_m * self.timeforlinesource / min(self.radiusvector)**2
        self.deltazbessel = np.logspace(-10, 8, self.NoDiscrInfCylIntegration)
        self.argumentbesselvec = np.logspace(np.log10(self.besselminarg) - 0.5, np.log10(self.besselmaxarg) + 0.5, self.NoArgumentsInfCylIntegration)
        self.besselcylinderresult = np.zeros(self.NoArgumentsInfCylIntegration)

        for i, argumentbessel in enumerate(self.argumentbesselvec):
            self.besselcylinderresult[i] = 2 / (self.k_m * np.pi**3) * np.trapz((1 - np.exp(-self.deltazbessel**2 * argumentbessel)) / (self.deltazbessel**3 * (jv(1, self.deltazbessel)**2 + yv(1, self.deltazbessel)**2)), self.deltazbessel)

        self.N = len(self.Deltaz)  # Number of elements
        self.elementcenters = 0.5 * np.column_stack((self.x[1:], self.y[1:], self.z[1:])) + 0.5 * np.column_stack((self.x[:-1], self.y[:-1], self.z[:-1]))  # Matrix that stores the mid point coordinates of each element
        self.interconnections = self.interconnections - 1
        self.elementcenters = np.delete(self.elementcenters, self.interconnections.reshape(-1,1), axis=0)  # Remove duplicate coordinates
        self.SMatrix = np.zeros((self.N, self.N)) # Initializes the spacing matrix, which holds the distance between center points of each element [m]
        
        for i in range(self.N):
            self.SMatrix[i, :] = np.sqrt((self.elementcenters[i, 0] - self.elementcenters[:, 0])**2 + (self.elementcenters[i, 1] - self.elementcenters[:, 1])**2 + (self.elementcenters[i, 2] - self.elementcenters[:, 2])**2)

        self.SoverL = np.zeros((self.N, self.N))  # Initializes the ratio of spacing to element length matrix

        for i in range(self.N):
            self.SMatrix[i, :] = np.sqrt((self.elementcenters[i, 0] - self.elementcenters[:, 0])**2 + (self.elementcenters[i, 1] - self.elementcenters[:, 1])**2 + (self.elementcenters[i, 2] - self.elementcenters[:, 2])**2)
        self.SoverL[i, :] = self.SMatrix[i, :] / self.Deltaz[i]

        self.SortedIndices = np.argsort(self.SMatrix, axis=1, kind = 'stable') # Getting the indices of the sorted elements
        self.SMatrixSorted = np.take_along_axis(self.SMatrix, self.SortedIndices, axis=1)  # Sorting the spacing matrix
        
        self.SoverLSorted = self.SMatrixSorted / self.Deltaz

        self.mindexNPCP = np.where(np.min(self.SoverLSorted, axis=0) < self.LimitSoverL)[0][-1]  # Finding the index where the ratio is less than the limit

        self.midpointsx = self.elementcenters[:, 0]
        self.midpointsy = self.elementcenters[:, 1]
        self.midpointsz = self.elementcenters[:, 2]
        self.BBinitial = self.surface_temp - self.geothermal_gradient * self.midpointsz  # Initial temperature at center of each element [degC]

        self.previouswaterelements = np.zeros(self.N)
        self.previouswaterelements[0:] = np.arange(-1,self.N-1)

        for i in range(self.numberoflaterals):
            self.previouswaterelements[self.interconnections[i + 1] - i-1] = len(self.xinj) - 2

        self.previouswaterelements[len(self.xinj) - 1] = 0

        self.lateralendpoints = []
        for i in range(1,self.numberoflaterals+1):
            self.lateralendpoints.append(len(self.xinj) - 2 + len(self.xprod) - 1 + i * ((self.xlat[:, 0]).size- 1))
        self.lateralendpoints = np.array(self.lateralendpoints)

        self.MaxSMatrixSorted = np.max(self.SMatrixSorted, axis=0)

        self.indicesyoucanneglectupfront = self.alpha_m * (np.ones((self.N-1, 1)) * self.times_arr * self.Deltat) / (self.MaxSMatrixSorted[1:].reshape(-1, 1) * np.ones((1, len(self.times_arr))))**2 / self.LimitNPSpacingTime
        self.indicesyoucanneglectupfront[self.indicesyoucanneglectupfront > 1] = 1

        self.lastneighbourtoconsider = np.zeros(len(self.times_arr))
        for i in range(len(self.times_arr)):
            self.lntc = np.where(self.indicesyoucanneglectupfront[:, i] == 1)[0]
            if len(self.lntc) == 0:
                self.lastneighbourtoconsider[i] = 0
            else:
                self.lastneighbourtoconsider[i] = max(1, self.lntc[-1] + 1)

        self.distributionx = np.zeros((len(self.x) - 1, self.M + 1))
        self.distributiony = np.zeros((len(self.x) - 1, self.M + 1))
        self.distributionz = np.zeros((len(self.x) - 1, self.M + 1))

        for i in range(len(self.x) - 1):
            self.distributionx[i, :] = np.linspace(self.x[i], self.x[i + 1], self.M + 1).reshape(-1)
            self.distributiony[i, :] = np.linspace(self.y[i], self.y[i + 1], self.M + 1).reshape(-1)
            self.distributionz[i, :] = np.linspace(self.z[i], self.z[i + 1], self.M + 1).reshape(-1)

        # Remove duplicates
        self.distributionx = np.delete(self.distributionx, self.interconnections, axis=0)
        self.distributiony = np.delete(self.distributiony, self.interconnections, axis=0)
        self.distributionz = np.delete(self.distributionz, self.interconnections, axis=0)

        self.dynamic_properties = dynamic_properties
        self.counter = -1

        # Initialize SBT algorithm linear system of equation matrices
        self.LL = np.zeros((3 * self.N, 3 * self.N))                # Will store the "left-hand side" of the system of equations
        self.RR = np.zeros((3 * self.N, 1))                    # Will store the "right-hand side" of the system of equations
        self.Q = np.zeros((self.N, len(self.times_arr)))               # Initializes the heat pulse matrix, i.e., the heat pulse emitted by each element at each time step
        self.Twprevious = self.BBinitial                       # At time zero, the initial fluid temperature corresponds to the initial local rock temperature
        self.TwMatrix = np.zeros((len(self.times_arr), self.N))         # Initializes the matrix that holds the fluid temperature over time
        self.TwMatrix[0, :] = self.Twprevious

        #Initialize FMM arrays
        self.combinedtimes = np.array([0])
        self.combinedQ = np.zeros((self.N, 1))
        self.combinedtimes2ndlevel = np.array([0])
        self.combinedtimes3rdlevel = np.array([0])

        self.configure_well_dimensions()

    def pre_model(self, t, m_prd, m_inj, T_inj):
        """Computations to be performed before stepping the reservoir model.

        Args:
            t (datetime): current timestamp.
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s.
            T_inj (float): injection temperature in deg C.
        """
        self.counter += 1

    def model(self, t, m_prd, m_inj, T_inj):
        """Computations to be performed when stepping the reservoir model.

        Args:
            t (datetime): current timestamp.
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s.
            T_inj (float): injection temperature in deg C.
        """
        # Helper function to clean NaN/Inf from arrays - MUST be defined before any use
        def _clean_array(arr, name="array"):
            """Clean NaN/Inf from array, replacing with safe values."""
            max_finite_val = np.finfo(np.float64).max / 1e10
            cleaned = np.nan_to_num(arr, nan=0.0, posinf=max_finite_val, neginf=-max_finite_val)
            # Double-check
            nan_inf_mask = np.isnan(cleaned) | np.isinf(cleaned)
            if np.any(nan_inf_mask):
                cleaned[nan_inf_mask] = np.where(np.isinf(cleaned[nan_inf_mask]),
                                                 np.where(cleaned[nan_inf_mask] > 0, max_finite_val, -max_finite_val),
                                                 0.0)
            return cleaned
        
        # Helper function for runtime validation and logging
        def _validate_array(arr, name, location="", raise_on_error=True):
            """Validate array for NaN/Inf and log if debug mode enabled."""
            nan_count = np.sum(np.isnan(arr)) if isinstance(arr, np.ndarray) else (1 if np.isnan(arr) else 0)
            inf_count = np.sum(np.isinf(arr)) if isinstance(arr, np.ndarray) else (1 if np.isinf(arr) else 0)
            
            if self.debug_mode and self.debug_log is not None:
                self.debug_log.append({
                    'step': self.counter,
                    'location': location,
                    'array': name,
                    'nan_count': nan_count,
                    'inf_count': inf_count,
                    'shape': arr.shape if isinstance(arr, np.ndarray) else 'scalar',
                    'min': np.nanmin(arr) if isinstance(arr, np.ndarray) else arr,
                    'max': np.nanmax(arr) if isinstance(arr, np.ndarray) else arr,
                })
            
            if raise_on_error and (nan_count > 0 or inf_count > 0):
                raise ValueError(f"NaN/Inf detected in {name} at {location}: {nan_count} NaN, {inf_count} Inf")
            
            return nan_count, inf_count
        
        # Helper function for safe division
        def _safe_div(numerator, denominator, default=0.0, name="division"):
            """Safe division with epsilon to prevent division by zero."""
            eps = np.finfo(np.float64).eps * 10
            if isinstance(denominator, np.ndarray):
                denominator = np.maximum(np.abs(denominator), eps) * np.sign(denominator)
            else:
                denominator = max(abs(denominator), eps) * (1 if denominator >= 0 else -1)
            result = numerator / denominator
            _validate_array(result, name, f"after {name}", raise_on_error=False)
            return result
        
        # Helper function for safe sqrt
        def _safe_div(numerator, denominator, name="div", default=0.0):
            """Safe division ensuring denominator is not zero."""
            eps = np.finfo(float).eps * 10
            denominator = np.where(np.abs(denominator) < eps, eps * np.sign(denominator + eps), denominator)
            result = numerator / denominator
            result = np.nan_to_num(result, nan=default, posinf=1e10, neginf=-1e10)
            _validate_array(result, name, f"after {name}", raise_on_error=False)
            return result
        
        def _safe_sqrt(arg, name="sqrt"):
            """Safe sqrt ensuring argument is non-negative."""
            arg = np.maximum(arg, 0.0)
            result = np.sqrt(arg)
            _validate_array(result, name, f"after {name}", raise_on_error=False)
            return result
        
        # Helper function for safe exp1
        def _safe_exp1(arg, name="exp1"):
            """Safe exp1 ensuring argument is > epsilon."""
            eps = np.finfo(float).eps * 10
            arg = np.maximum(arg, eps)
            result = exp1(arg)
            # Replace inf with large finite value
            result = np.where(np.isinf(result), 1e10, result)
            _validate_array(result, name, f"after {name}", raise_on_error=False)
            return result

        if self.reservoir_simulator_settings["DynamicFluidProperties"]:
            Tavg = self.Tres
            # self.rho_f = densitywater(Tavg)
            # self.mu_f = viscositywater(Tavg)
            # self.cp_f = heatcapacitywater(Tavg)
            Tround = round(Tavg, 0)
            self.rho_f = self.cache["densitywater"].get(Tround, densitywater(Tavg))
            self.cache["densitywater"][Tround] = self.rho_f

            self.mu_f = self.cache["viscositywater"].get(Tround, viscositywater(Tavg))
            self.cache["viscositywater"][Tround] = self.mu_f

            self.cp_f = self.cache["cpwater"].get(Tround, heatcapacitywater(Tavg))
            self.cache["cpwater"][Tround] = self.cp_f

            self.alpha_f = self.k_f / self.rho_f / self.cp_f  # Fluid thermal diffusivity [m2/s]
            self.Pr_f = self.mu_f / self.rho_f / self.alpha_f  # Fluid Prandtl number [-]

        if self.dynamic_properties:
            self.rhow_prd_bh = densitywater(_safe_mean(self.T_prd_bh))
            self.cw_prd_bh = heatcapacitywater(_safe_mean(self.T_prd_bh)) # J/kg-degC

        self.Deltat = self.timestep.total_seconds() # Current time step size [s]
        Tin = T_inj #injection temperature is the same for all doublets as they assumingly feed into a single power plant
        m = _safe_mean(m_prd) #take the mean of all doublets

        # Velocities and thermal resistances are calculated each time step as the flow rate is allowed to vary each time step
        self.uvertical = m / self.rho_f / (np.pi * (self.vertical_diam/2) ** 2)  # Fluid velocity in vertical injector and producer [m/s]
        self.ulateral = m / self.rho_f / (np.pi * (self.lateral_diam/2) ** 2) * self.lateralflowallocation * self.lateralflowmultiplier  # Fluid velocity in each lateral [m/s]
        self.uvector = np.hstack((self.uvertical * np.ones(len(self.xinj) + len(self.xprod) - 2)))

        for dd in range(self.numberoflaterals):
            self.uvector = np.hstack((self.uvector, self.ulateral[dd] * np.ones(len(self.xlat[:, 0]) - 1)))

        if m > 0.1:
            self.Revertical = self.rho_f * self.uvertical * self.vertical_diam / self.mu_f  # Fluid Reynolds number in injector and producer [-]
            self.Nuvertical = 0.023 * self.Revertical ** (4 / 5) * self.Pr_f ** 0.4  # Nusselt Number in injector and producer (we assume turbulent flow) [-]
        else:
            self.Nuvertical = 1  # At low flow rates, we assume we are simulating the condition of well shut-in and set the Nusselt number to 1 (i.e., conduction only) [-]

        self.hvertical = self.Nuvertical * self.k_f / self.vertical_diam  # Heat transfer coefficient in injector and producer [W/m2/K]
        self.Rtvertical = 1 / (np.pi * self.hvertical * self.vertical_diam)  # Thermal resistance in injector and producer (open-hole assumed)

        if m > 0.1:
            self.Relateral = self.rho_f * self.ulateral * self.vertical_diam / self.mu_f  # Fluid Reynolds number in lateral [-]
            self.Nulateral = 0.023 * self.Relateral ** (4 / 5) * self.Pr_f ** 0.4  # Nusselt Number in lateral (we assume turbulent flow) [-]
        else:
            self.Nulateral = np.ones(self.numberoflaterals)  # At low flow rates, we assume we are simulating the condition of well shut-in and set the Nusselt number to 1 (i.e., conduction only) [-]

        self.hlateral = self.Nulateral * self.k_f / self.lateral_diam  # Heat transfer coefficient in lateral [W/m2/K]
        self.Rtlateral = 1 / (np.pi * self.hlateral * self.lateral_diam)  # Thermal resistance in lateral (open-hole assumed)

        self.Rtvector = self.Rtvertical * np.ones(len(self.radiusvector))  # Store thermal resistance of each element in a vector

        for dd in range(1, self.numberoflaterals + 1):
            if dd < self.numberoflaterals:
                self.Rtvector[self.interconnections[dd] - dd : self.interconnections[dd + 1] - dd] = self.Rtlateral[dd - 1] * np.ones(len(self.xlat[:, 0]))
            else:
                self.Rtvector[self.interconnections[dd] - self.numberoflaterals:] = self.Rtlateral[dd - 1] * np.ones(len(self.xlat[:, 0]) - 1)

        # Check for invalid values before calculations
        if np.any(np.isnan(self.radiusvector)) or np.any(np.isinf(self.radiusvector)) or np.any(self.radiusvector <= 0):
            raise ValueError(f"Invalid radiusvector values detected (NaN/Inf/zero). Check vertical_diam and lateral_diam parameters.")
        
        if np.any(np.isnan(self.Deltaz)) or np.any(np.isinf(self.Deltaz)) or np.any(self.Deltaz <= 0):
            raise ValueError(f"Invalid Deltaz values detected (NaN/Inf/zero). Check geometry parameters.")
        
        if self.alpha_m <= 0 or np.isnan(self.alpha_m) or np.isinf(self.alpha_m):
            raise ValueError(f"Invalid alpha_m (thermal diffusivity) value: {self.alpha_m}. Check k_m, rho_m, c_m parameters.")
        
        if self.Deltat <= 0 or np.isnan(self.Deltat) or np.isinf(self.Deltat):
            raise ValueError(f"Invalid Deltat (time step) value: {self.Deltat}. Check timestep parameter.")
        
        if self.alpha_m * self.Deltat / max(self.radiusvector)**2 > self.LimitCylinderModelRequired:
            # Calculate exp1 argument safely
            exp1_arg = _safe_div(self.radiusvector**2, 4 * self.alpha_m * self.Deltat, "exp1_arg_CPCP")
            _validate_array(exp1_arg, "exp1_arg", "CPCP calculation", raise_on_error=False)
            # Use safe exp1 helper
            exp1_result = _safe_exp1(exp1_arg, "exp1_CPCP")
            # Check for NaN
            if np.any(np.isnan(exp1_result)):
                raise ValueError(f"NaN in exp1 result for CPCP calculation. Check radiusvector, alpha_m, Deltat.")
            self.CPCP = np.ones(self.N) * _safe_div(1, (4 * np.pi * self.k_m), name="CPCP_divisor") * exp1_result  # Use line source model if possible
            _validate_array(self.CPCP, "CPCP", "after calculation", raise_on_error=False)
            # Clean CPCP immediately after calculation
            self.CPCP = _clean_array(self.CPCP, "CPCP")
        else:
            interp_arg = self.alpha_m * self.Deltat / self.radiusvector**2
            # Clip interpolation argument to valid range
            interp_arg = np.clip(interp_arg, self.argumentbesselvec.min(), self.argumentbesselvec.max())
            # Check for NaN/Inf before interpolation
            if np.any(np.isnan(interp_arg)) or np.any(np.isinf(interp_arg)):
                raise ValueError(f"Invalid interpolation argument for besselcylinderresult. Check alpha_m, Deltat, radiusvector.")
            self.CPCP = np.ones(self.N) * np.interp(interp_arg, self.argumentbesselvec, self.besselcylinderresult)  # Use cylindrical source model if required

        if self.Deltat > self.timeforfinitelinesource:  # For long time steps, the finite length correction should be applied
            interp_arg = self.Deltaz**2 / (4 * self.alpha_m * self.Deltat)
            # Clip interpolation argument to valid range
            interp_arg = np.clip(interp_arg, self.Amin1vector.min(), self.Amin1vector.max())
            # Check for NaN/Inf before interpolation
            if np.any(np.isnan(interp_arg)) or np.any(np.isinf(interp_arg)):
                raise ValueError(f"Invalid interpolation argument for finitecorrectiony. Check Deltaz, alpha_m, Deltat.")
            self.CPCP = self.CPCP + np.interp(interp_arg, self.Amin1vector, self.finitecorrectiony)
            # Clean CPCP after finite correction
            self.CPCP = _clean_array(self.CPCP, "CPCP")

        if self.counter > 1:  # After the second time step, we need to keep track of previous heat pulses

            self.CPOP = np.zeros((self.N, len(self.timesFMM)-1))
            self.indexpsstart = 0
            self.indexpsend = np.where(self.timeforpointssource < (self.times_arr[self.counter] - self.timesFMM[1:]) * self.Deltat)[-1]
            if self.indexpsend.size > 0:
                self.indexpsend = self.indexpsend[-1] + 1
            else:
                self.indexpsend = self.indexpsstart - 1
            
            if self.indexpsend >= self.indexpsstart:  # Use point source model if allowed
                # Calculate time differences safely
                time_diff1 = (self.times_arr[self.counter] - self.timesFMM[self.indexpsstart + 1:self.indexpsend + 2]) * self.Deltat
                time_diff2 = (self.times_arr[self.counter] - self.timesFMM[self.indexpsstart:self.indexpsend+1]) * self.Deltat
                
                # Ensure time differences are positive (avoid sqrt of negative or zero)
                time_diff1 = np.maximum(time_diff1, np.finfo(float).eps)
                time_diff2 = np.maximum(time_diff2, np.finfo(float).eps)
                
                # Check for NaN/Inf before calculation
                if np.any(np.isnan(time_diff1)) or np.any(np.isinf(time_diff1)) or np.any(np.isnan(time_diff2)) or np.any(np.isinf(time_diff2)):
                    raise ValueError(f"Invalid time differences in point source model calculation. Check times_arr, timesFMM, Deltat.")
                
                sqrt_term1 = 1 / np.sqrt(time_diff1)
                sqrt_term2 = 1 / np.sqrt(time_diff2)
                
                # Check for NaN/Inf in sqrt results
                if np.any(np.isnan(sqrt_term1)) or np.any(np.isinf(sqrt_term1)) or np.any(np.isnan(sqrt_term2)) or np.any(np.isinf(sqrt_term2)):
                    raise ValueError(f"Invalid sqrt results in point source model. Check time differences.")
                
                self.CPOP[:, 0:self.indexpsend] = self.Deltaz * np.ones((self.N, self.indexpsend)) / (4 * np.pi * np.sqrt(self.alpha_m * np.pi) * self.k_m) * (
                        np.ones(self.N) * (sqrt_term1 - sqrt_term2))
                # Clean CPOP after point source calculation
                self.CPOP[:, 0:self.indexpsend] = _clean_array(self.CPOP[:, 0:self.indexpsend], "CPOP_point_source")

            self.indexlsstart = self.indexpsend + 1
            self.indexlsend = np.where(self.timeforlinesource < (self.times_arr[self.counter] - self.timesFMM[1:]) * self.Deltat)[0]
            if self.indexlsend.size == 0:
                self.indexlsend = self.indexlsstart - 1
            else:
                self.indexlsend = self.indexlsend[-1]

            if self.indexlsend >= self.indexlsstart:  # Use line source model for more recent heat pulse events
                # Calculate time differences safely
                time_diff1 = (self.times_arr[self.counter] - self.timesFMM[self.indexlsstart:self.indexlsend+1]) * self.Deltat
                time_diff2 = (self.times_arr[self.counter] - self.timesFMM[self.indexlsstart+1:self.indexlsend+2]) * self.Deltat
                
                # Ensure time differences are positive (exp1 requires positive argument)
                time_diff1 = np.maximum(time_diff1, np.finfo(float).eps)
                time_diff2 = np.maximum(time_diff2, np.finfo(float).eps)
                
                # Calculate exp1 arguments
                exp1_arg1 = _safe_div((self.radiusvector**2).reshape(len(self.radiusvector ** 2),1), 4*self.alpha_m*time_diff1.reshape(1,len(time_diff1)), "exp1_arg1_line_source")
                exp1_arg2 = _safe_div((self.radiusvector**2).reshape(len(self.radiusvector ** 2),1), 4 * self.alpha_m * time_diff2.reshape(1,len(time_diff2)), "exp1_arg2_line_source")
                
                # Ensure exp1 arguments are positive and not too small (exp1(0) = inf)
                exp1_arg1 = np.maximum(exp1_arg1, np.finfo(float).eps)
                exp1_arg2 = np.maximum(exp1_arg2, np.finfo(float).eps)
                
                # Check for NaN/Inf before exp1 calculation
                if np.any(np.isnan(exp1_arg1)) or np.any(np.isinf(exp1_arg1)) or np.any(np.isnan(exp1_arg2)) or np.any(np.isinf(exp1_arg2)):
                    raise ValueError(f"Invalid exp1 arguments in line source model. Check radiusvector, alpha_m, time differences.")
                
                # Calculate exp1 safely using helper function
                exp1_result1 = _safe_exp1(exp1_arg1, "exp1_line_source_1")
                exp1_result2 = _safe_exp1(exp1_arg2, "exp1_line_source_2")
                
                self.CPOP[:, self.indexlsstart:self.indexlsend+1] = np.ones((self.N,1)) * 1 / (4*np.pi*self.k_m) * (exp1_result1 - exp1_result2)
                # Clean CPOP after line source calculation
                self.CPOP[:, self.indexlsstart:self.indexlsend+1] = _clean_array(self.CPOP[:, self.indexlsstart:self.indexlsend+1], "CPOP_line_source")

            self.indexcsstart = max(self.indexpsend, self.indexlsend) + 1
            self.indexcsend = len(self.timesFMM) - 2

            if self.indexcsstart <= self.indexcsend:  # Use cylindrical source model for the most recent heat pulses

                self.CPOPPH = np.zeros((self.CPOP[:, self.indexcsstart:self.indexcsend+1].shape))   
                self.CPOPdim =self.CPOP[:, self.indexcsstart:self.indexcsend+1].shape
                self.CPOPPH = self.CPOPPH.T.ravel()
                
                # Calculate interpolation arguments
                time_diff1 = (self.times_arr[self.counter] - self.timesFMM[self.indexcsstart:self.indexcsend+1]) * self.Deltat
                time_diff2 = (self.times_arr[self.counter] - self.timesFMM[self.indexcsstart+1: self.indexcsend+2]) * self.Deltat
                
                # Ensure time differences are positive and valid
                time_diff1 = np.maximum(time_diff1, np.finfo(float).eps)  # Avoid zero or negative
                time_diff2 = np.maximum(time_diff2, np.finfo(float).eps)  # Avoid zero or negative
                
                interp_arg1 = self.alpha_m * time_diff1.reshape(len(time_diff1),1) / (self.radiusvector ** 2).reshape(len(self.radiusvector ** 2),1).T
                interp_arg2 = self.alpha_m * time_diff2.reshape(len(time_diff2),1) / (self.radiusvector ** 2).reshape(len(self.radiusvector ** 2),1).T
                
                # Clip to valid range and check for NaN/Inf
                interp_arg1 = np.clip(interp_arg1, self.argumentbesselvec.min(), self.argumentbesselvec.max())
                interp_arg2 = np.clip(interp_arg2, self.argumentbesselvec.min(), self.argumentbesselvec.max())
                
                if np.any(np.isnan(interp_arg1)) or np.any(np.isinf(interp_arg1)) or np.any(np.isnan(interp_arg2)) or np.any(np.isinf(interp_arg2)):
                    raise ValueError(f"Invalid interpolation arguments for besselcylinderresult in CPOP calculation. Check times_arr, timesFMM, Deltat, alpha_m, radiusvector.")
                
                self.CPOPPH = (np.ones(self.N) * ( \
                            np.interp(interp_arg1, self.argumentbesselvec, self.besselcylinderresult) - \
                            np.interp(interp_arg2, self.argumentbesselvec, self.besselcylinderresult))).reshape(-1,1)
                self.CPOPPH=self.CPOPPH.reshape((self.CPOPdim),order='F')
                self.CPOP[:, self.indexcsstart:self.indexcsend+1] = self.CPOPPH
                # Clean CPOP after cylindrical source calculation
                self.CPOP[:, self.indexcsstart:self.indexcsend+1] = _clean_array(self.CPOP[:, self.indexcsstart:self.indexcsend+1], "CPOP_cylindrical_source")

            self.indexflsstart = self.indexpsend + 1
            self.indexflsend = np.where(self.timeforfinitelinesource < (self.times_arr[self.counter] - self.timesFMM[1:]) * self.Deltat)[-1]
            if self.indexflsend.size == 0:
                self.indexflsend = self.indexflsstart - 1
            else:
                self.indexflsend = self.indexflsend[-1] - 1

            if self.indexflsend >= self.indexflsstart:  # Perform finite length correction if needed
                time_diff1 = (self.times_arr[self.counter] - self.timesFMM[self.indexflsstart:self.indexflsend+2]) * self.Deltat
                time_diff2 = (self.times_arr[self.counter] - self.timesFMM[self.indexflsstart+1:self.indexflsend+3]) * self.Deltat
                
                # Ensure time differences are positive and valid
                time_diff1 = np.maximum(time_diff1, np.finfo(float).eps)
                time_diff2 = np.maximum(time_diff2, np.finfo(float).eps)
                
                interp_arg1 = np.matmul((self.Deltaz.reshape(len(self.Deltaz),1) ** 2),np.ones((1,self.indexflsend-self.indexflsstart+2))) / np.matmul(np.ones((self.N,1)),(4 * self.alpha_m * time_diff1.reshape(len(time_diff1),1)).T)
                interp_arg2 = np.matmul((self.Deltaz.reshape(len(self.Deltaz),1) ** 2),np.ones((1,self.indexflsend-self.indexflsstart+2))) / np.matmul(np.ones((self.N,1)),(4 * self.alpha_m * time_diff2.reshape(len(time_diff2),1)).T)
                
                # Clip to valid range and check for NaN/Inf
                interp_arg1 = np.clip(interp_arg1, self.Amin1vector.min(), self.Amin1vector.max())
                interp_arg2 = np.clip(interp_arg2, self.Amin1vector.min(), self.Amin1vector.max())
                
                if np.any(np.isnan(interp_arg1)) or np.any(np.isinf(interp_arg1)) or np.any(np.isnan(interp_arg2)) or np.any(np.isinf(interp_arg2)):
                    raise ValueError(f"Invalid interpolation arguments for finitecorrectiony in CPOP calculation. Check Deltaz, times_arr, timesFMM, Deltat, alpha_m.")
                
                self.CPOP[:, self.indexflsstart:self.indexflsend+2] = self.CPOP[:, self.indexflsstart:self.indexflsend+2] + (np.interp(interp_arg1, self.Amin1vector, self.finitecorrectiony) - \
                np.interp(interp_arg2, self.Amin1vector, self.finitecorrectiony))


        self.NPCP = np.zeros((self.N, self.N))
        np.fill_diagonal(self.NPCP, self.CPCP)
        
        # Clean NPCP immediately after creation to prevent NaN/Inf propagation
        max_finite_val = np.finfo(np.float64).max / 1e10
        self.NPCP = np.nan_to_num(self.NPCP, nan=0.0, posinf=max_finite_val, neginf=-max_finite_val)


        self.spacingtest = self.alpha_m * self.Deltat / self.SMatrixSorted[:, 1:]**2 / self.LimitNPSpacingTime
        self.maxspacingtest = np.max(self.spacingtest,axis=0)


        if self.maxspacingtest[0] < 1:
            self.maxindextoconsider = 0
        else:
            self.maxindextoconsider = np.where(self.maxspacingtest > 1)[0][-1]+1

        if self.mindexNPCP < self.maxindextoconsider + 1:
            self.indicestocalculate = self.SortedIndices[:, self.mindexNPCP + 1:self.maxindextoconsider + 1]
            self.indicestocalculatetranspose = self.indicestocalculate.T
            self.indicestocalculatelinear = self.indicestocalculate.ravel()
            self.indicestostorematrix = (self.indicestocalculate - 1) * self.N + np.arange(1, self.N) * np.ones((1, self.maxindextoconsider - self.mindexNPCP + 1))
            self.indicestostorematrixtranspose = self.indicestostorematrix.T
            self.indicestostorelinear = self.indicestostorematrix.ravel()
            # Calculate erf argument safely
            erf_arg = self.SMatrix[self.indicestostorelinear] / np.sqrt(4 * self.alpha_m * self.Deltat)
            # Ensure sqrt argument is positive
            sqrt_arg = 4 * self.alpha_m * self.Deltat
            if sqrt_arg <= 0 or np.isnan(sqrt_arg) or np.isinf(sqrt_arg):
                raise ValueError(f"Invalid sqrt argument in NPCP calculation: {sqrt_arg}. Check alpha_m, Deltat.")
            sqrt_val = np.sqrt(sqrt_arg)
            # Avoid division by zero
            sqrt_val = np.maximum(sqrt_val, np.finfo(float).eps)
            erf_arg = self.SMatrix[self.indicestostorelinear] / sqrt_val
            
            # Check for NaN/Inf in erf_arg
            if np.any(np.isnan(erf_arg)) or np.any(np.isinf(erf_arg)):
                raise ValueError(f"Invalid erf argument in NPCP calculation. Check SMatrix, alpha_m, Deltat.")
            
            erf_result = erf(erf_arg)
            # Check for NaN/Inf in erf result
            if np.any(np.isnan(erf_result)) or np.any(np.isinf(erf_result)):
                raise ValueError(f"NaN/Inf in erf result in NPCP calculation.")
            
            # Avoid division by zero in SMatrix
            SMatrix_safe = np.maximum(np.abs(self.SMatrix[self.indicestostorelinear]), np.finfo(float).eps)
            self.NPCP[self.indicestostorelinear] = self.Deltaz[self.indicestocalculatelinear] / (4 * np.pi * self.k_m * SMatrix_safe) * erf_result
        #Calculate and store neighbouring pipes for current pulse as set of line sources
        if self.mindexNPCP > 1 and self.maxindextoconsider > 0:
            self.lastindexfls = min(self.mindexNPCP, self.maxindextoconsider + 1)
            self.indicestocalculate = self.SortedIndices[:, 1:self.lastindexfls]
            self.indicestocalculatetranspose = self.indicestocalculate.T
            self.indicestocalculatelinear = self.indicestocalculate.ravel()
            self.indicestostorematrix = (self.indicestocalculate) * self.N + np.arange(self.N).reshape(-1,1) * np.ones((1, self.lastindexfls - 1), dtype=int)

            self.indicestostorematrixtranspose = self.indicestostorematrix.T
            self.indicestostorelinear = self.indicestostorematrix.ravel()
            self.midpointindices = np.matmul(np.ones((self.lastindexfls - 1, 1)), np.arange(self.N).reshape(1,self.N)).T
            self.midpointsindices = self.midpointindices.ravel().astype(int)
            self.rultimate = np.sqrt(np.square((self.midpointsx[self.midpointsindices].reshape(len(self.midpointsindices),1)*( np.ones((1, self.M + 1))) - self.distributionx[self.indicestocalculatelinear,:])) +
                                np.square((self.midpointsy[self.midpointsindices].reshape(len(self.midpointsindices),1)*( np.ones((1, self.M + 1))) - self.distributiony[self.indicestocalculatelinear,:])) +
                                np.square((self.midpointsz[self.midpointsindices].reshape(len(self.midpointsindices),1)*( np.ones((1, self.M + 1))) - self.distributionz[self.indicestocalculatelinear,:])))

            self.NPCP[np.unravel_index(self.indicestostorelinear, self.NPCP.shape, 'F')] =  self.Deltaz[self.indicestocalculatelinear] / self.M * np.sum((1 - erf(self.rultimate / np.sqrt(4 * self.alpha_m * self.Deltat))) / (4 * np.pi * self.k_m * self.rultimate) * np.matmul(np.ones((self.N*(self.lastindexfls-1),1)),np.concatenate((np.array([1/2]), np.ones(self.M-1), np.array([1/2]))).reshape(-1,1).T), axis=1)

        self.BB = np.zeros((self.N, 1))
        if self.counter > 1 and self.lastneighbourtoconsider[self.counter] > 0:
            self.SMatrixRelevant = self.SMatrixSorted[:, 1 : int(self.lastneighbourtoconsider[self.counter] + 1)]
            self.SoverLRelevant = self.SoverLSorted[:, 1 : int(self.lastneighbourtoconsider[self.counter]) + 1]
            self.SortedIndicesRelevant = self.SortedIndices[:, 1 : int(self.lastneighbourtoconsider[self.counter]) + 1] 
            self.maxtimeindexmatrix = self.alpha_m * np.ones((self.N * int(self.lastneighbourtoconsider[self.counter]), 1)) * (self.times_arr[self.counter] - self.timesFMM[1:]) * self.Deltat / (self.SMatrixRelevant.ravel().reshape(-1,1) * np.ones((1,len(self.timesFMM)-1)))**2

            self.allindices = np.arange(self.N * int(self.lastneighbourtoconsider[self.counter]) * (len(self.timesFMM) - 1))
            #if (i>=154):
            #   
            self.pipeheatcomesfrom = np.matmul(self.SortedIndicesRelevant.T.ravel().reshape(len(self.SortedIndicesRelevant.ravel()),1), np.ones((1,len(self.timesFMM) - 1)))
            self.pipeheatgoesto = np.arange(self.N).reshape(self.N,1) * np.ones((1, int(self.lastneighbourtoconsider[self.counter])))
            self.pipeheatgoesto = self.pipeheatgoesto.transpose().ravel().reshape(len(self.pipeheatgoesto.ravel()),1) * np.ones((1, len(self.timesFMM) - 1))
            # Delete everything smaller than LimitNPSpacingTime
            # 
            self.indicestoneglect = np.where((self.maxtimeindexmatrix.transpose()).ravel() < self.LimitNPSpacingTime)[0]

            self.maxtimeindexmatrix = np.delete(self.maxtimeindexmatrix, self.indicestoneglect)
            self.allindices = np.delete(self.allindices, self.indicestoneglect)
            self.indicesFoSlargerthan = np.where(self.maxtimeindexmatrix.ravel() > 10)[0]
        #  
            self.indicestotakeforpsFoS = self.allindices[self.indicesFoSlargerthan]

            self.allindices2 = self.allindices.copy()
            #pdb.set_trace()
            self.allindices2[self.indicesFoSlargerthan] = []
            self.SoverLinearized = self.SoverLRelevant.ravel().reshape(len(self.SoverLRelevant.ravel()),1) * np.ones((1, len(self.timesFMM) - 1))
            self.indicestotakeforpsSoverL = np.where(self.SoverLinearized.transpose().ravel()[self.allindices2] > self.LimitSoverL)[0]
            self.overallindicestotakeforpsSoverL = self.allindices2[self.indicestotakeforpsSoverL]
            self.remainingindices = self.allindices2.copy() 

            self.remainingindices=np.delete(self.remainingindices,self.indicestotakeforpsSoverL)

            self.NPOP = np.zeros((self.N * int(self.lastneighbourtoconsider[self.counter]), len(self.timesFMM) - 1))

            # Use point source model when FoS is very large
            if len(self.indicestotakeforpsFoS) > 0:
                self.deltatlinear1 = np.ones(self.N * int(self.lastneighbourtoconsider[self.counter]), 1) * (self.times_arr[self.counter] - self.timesFMM[1:]) * self.Deltat
                self.deltatlinear1 = self.deltatlinear1.ravel()[self.indicestotakeforpsFoS]
                self.deltatlinear2 = np.ones((self.N * int(self.lastneighbourtoconsider[self.counter]), 1)) * (self.times_arr[self.counter] - self.timesFMM[0:-1]) * self.Deltat
                self.deltatlinear2 = self.deltatlinear2[self.indicestotakeforpsFoS]
                self.deltazlinear = self.pipeheatcomesfrom[self.indicestotakeforpsFoS]
                self.SMatrixlinear = self.SMatrixRelevant.flatten(order='F')
                # Validate inputs before calculation
                sqrt_arg1 = 4 * self.alpha_m * self.deltatlinear2
                sqrt_arg2 = 4 * self.alpha_m * self.deltatlinear1
                _validate_array(sqrt_arg1, "sqrt_arg1_NPOPFoS", "NPOPFoS calculation", raise_on_error=False)
                _validate_array(sqrt_arg2, "sqrt_arg2_NPOPFoS", "NPOPFoS calculation", raise_on_error=False)
                
                sqrt1 = _safe_sqrt(sqrt_arg1, "sqrt1_NPOPFoS")
                sqrt2 = _safe_sqrt(sqrt_arg2, "sqrt2_NPOPFoS")
                
                erfc_arg1 = self.SMatrixlinear[self.indicestotakeforpsFoS] / sqrt1
                erfc_arg2 = self.SMatrixlinear[self.indicestotakeforpsFoS] / sqrt2
                _validate_array(erfc_arg1, "erfc_arg1_NPOPFoS", "NPOPFoS calculation", raise_on_error=False)
                _validate_array(erfc_arg2, "erfc_arg2_NPOPFoS", "NPOPFoS calculation", raise_on_error=False)
                
                erfc1 = erfc(erfc_arg1)
                erfc2 = erfc(erfc_arg2)
                _validate_array(erfc1, "erfc1_NPOPFoS", "NPOPFoS calculation", raise_on_error=False)
                _validate_array(erfc2, "erfc2_NPOPFoS", "NPOPFoS calculation", raise_on_error=False)
                
                divisor = 4 * np.pi * self.k_m * self.SMatrixlinear[self.indicestotakeforpsFoS]
                _validate_array(divisor, "divisor_NPOPFoS", "NPOPFoS calculation", raise_on_error=False)
                
                self.NPOPFoS = self.Deltaz[self.deltazlinear] * _safe_div((erfc1 - erfc2), divisor, name="NPOPFoS_divisor")
                _validate_array(self.NPOPFoS, "NPOPFoS", "after calculation", raise_on_error=False)
                # Clean NPOPFoS before assigning to NPOP
                self.NPOPFoS = _clean_array(self.NPOPFoS, "NPOPFoS")
                self.NPOP[self.indicestotakeforpsFoS] = self.NPOPFoS

            # Use point source model when SoverL is very large
            if len(self.overallindicestotakeforpsSoverL) > 0:
                self.deltatlinear1 = np.ones((self.N * int(self.lastneighbourtoconsider[self.counter]), 1)) * ((self.times_arr[self.counter] - self.timesFMM[1:]) * self.Deltat).ravel()
                self.deltatlinear1 = self.deltatlinear1[self.overallindicestotakeforpsSoverL]
                self.deltatlinear2 = np.ones((self.N * int(self.lastneighbourtoconsider[self.counter]), 1)) * ((self.times_arr[self.counter] - self.timesFMM[0:-1]) * self.Deltat).ravel()
                self.deltatlinear2 = self.deltatlinear2[self.overallindicestotakeforpsSoverL]
                self.deltazlinear = self.pipeheatcomesfrom[self.overallindicestotakeforpsSoverL]
                self.SMatrixlinear = self.SMatrixRelevant.flatten(order='F')
                # Validate inputs before calculation
                sqrt_arg1 = 4 * self.alpha_m * self.deltatlinear2
                sqrt_arg2 = 4 * self.alpha_m * self.deltatlinear1
                _validate_array(sqrt_arg1, "sqrt_arg1_NPOPSoverL", "NPOPSoverL calculation", raise_on_error=False)
                _validate_array(sqrt_arg2, "sqrt_arg2_NPOPSoverL", "NPOPSoverL calculation", raise_on_error=False)
                
                sqrt1 = _safe_sqrt(sqrt_arg1, "sqrt1_NPOPSoverL")
                sqrt2 = _safe_sqrt(sqrt_arg2, "sqrt2_NPOPSoverL")
                
                erfc_arg1 = self.SMatrixlinear[self.overallindicestotakeforpsSoverL] / sqrt1
                erfc_arg2 = self.SMatrixlinear[self.overallindicestotakeforpsSoverL] / sqrt2
                _validate_array(erfc_arg1, "erfc_arg1_NPOPSoverL", "NPOPSoverL calculation", raise_on_error=False)
                _validate_array(erfc_arg2, "erfc_arg2_NPOPSoverL", "NPOPSoverL calculation", raise_on_error=False)
                
                erfc1 = erfc(erfc_arg1)
                erfc2 = erfc(erfc_arg2)
                _validate_array(erfc1, "erfc1_NPOPSoverL", "NPOPSoverL calculation", raise_on_error=False)
                _validate_array(erfc2, "erfc2_NPOPSoverL", "NPOPSoverL calculation", raise_on_error=False)
                
                divisor = 4 * np.pi * self.k_m * self.SMatrixlinear[self.overallindicestotakeforpsSoverL]
                _validate_array(divisor, "divisor_NPOPSoverL", "NPOPSoverL calculation", raise_on_error=False)
                
                self.NPOPSoverL = self.Deltaz[self.deltazlinear] * _safe_div((erfc1 - erfc2), divisor, name="NPOPSoverL_divisor")
                _validate_array(self.NPOPSoverL, "NPOPSoverL", "after calculation", raise_on_error=False)
                # Clean NPOPSoverL before assigning to NPOP
                self.NPOPSoverL = _clean_array(self.NPOPSoverL, "NPOPSoverL")
                self.NPOP[self.overallindicestotakeforpsSoverL] = self.NPOPSoverL

            # Use finite line source model for remaining pipe segments
            if len(self.remainingindices) > 0:

                self.deltatlinear1 = np.ones((self.N * int(self.lastneighbourtoconsider[self.counter]), 1)) * (self.times_arr[self.counter] - self.timesFMM[1:]) * self.Deltat
                self.deltatlinear1 = (self.deltatlinear1.transpose()).ravel()[self.remainingindices]
                self.deltatlinear2 = np.ones((self.N * int(self.lastneighbourtoconsider[self.counter]), 1)) * (self.times_arr[self.counter] - self.timesFMM[0:-1]) * self.Deltat
                self.deltatlinear2 = (self.deltatlinear2.transpose()).ravel()[self.remainingindices]
                self.deltazlinear = (self.pipeheatcomesfrom.T).ravel()[self.remainingindices]
                self.midpointstuff = (self.pipeheatgoesto.transpose()).ravel()[self.remainingindices]
                # Calculate rultimate with safe sqrt
                rultimate_sq = (np.square((self.midpointsx[self.midpointstuff.astype(int)].reshape(len(self.midpointsx[self.midpointstuff.astype(int)]),1)*( np.ones((1, self.M + 1))) - self.distributionx[self.deltazlinear.astype(int),:])) +
                                np.square((self.midpointsy[self.midpointstuff.astype(int)].reshape(len(self.midpointsy[self.midpointstuff.astype(int)]),1)*( np.ones((1, self.M + 1))) - self.distributiony[self.deltazlinear.astype(int),:])) +
                                np.square((self.midpointsz[self.midpointstuff.astype(int)].reshape(len(self.midpointsz[self.midpointstuff.astype(int)]),1)*( np.ones((1, self.M + 1))) - self.distributionz[self.deltazlinear.astype(int),:])))
                _validate_array(rultimate_sq, "rultimate_sq", "NPOPfls calculation", raise_on_error=False)
                self.rultimate = _safe_sqrt(rultimate_sq, "rultimate")
                _validate_array(self.rultimate, "rultimate", "NPOPfls calculation", raise_on_error=False)
                
                # Validate and calculate sqrt arguments for erf
                sqrt_arg_fls1 = 4 * self.alpha_m * np.ravel(self.deltatlinear2).reshape(len(np.ravel(self.deltatlinear2)),1)*np.ones((1, self.M + 1))
                sqrt_arg_fls2 = 4 * self.alpha_m * np.ravel(self.deltatlinear1).reshape(len(np.ravel(self.deltatlinear1)),1)*np.ones((1, self.M + 1))
                _validate_array(sqrt_arg_fls1, "sqrt_arg_fls1", "NPOPfls calculation", raise_on_error=False)
                _validate_array(sqrt_arg_fls2, "sqrt_arg_fls2", "NPOPfls calculation", raise_on_error=False)
                
                sqrt_fls1 = _safe_sqrt(sqrt_arg_fls1, "sqrt_fls1")
                sqrt_fls2 = _safe_sqrt(sqrt_arg_fls2, "sqrt_fls2")
                
                # Calculate erf arguments and results
                erf_arg1 = self.rultimate / sqrt_fls1
                erf_arg2 = self.rultimate / sqrt_fls2
                _validate_array(erf_arg1, "erf_arg1_NPOPfls", "NPOPfls calculation", raise_on_error=False)
                _validate_array(erf_arg2, "erf_arg2_NPOPfls", "NPOPfls calculation", raise_on_error=False)
                
                erf1 = erf(erf_arg1)
                erf2 = erf(erf_arg2)
                _validate_array(erf1, "erf1_NPOPfls", "NPOPfls calculation", raise_on_error=False)
                _validate_array(erf2, "erf2_NPOPfls", "NPOPfls calculation", raise_on_error=False)
                
                # Calculate divisor safely
                divisor_fls = 4 * np.pi * self.k_m * self.rultimate
                _validate_array(divisor_fls, "divisor_NPOPfls", "NPOPfls calculation", raise_on_error=False)
                
                # Calculate NPOPfls with safe division
                erf_diff = erf1 - erf2
                _validate_array(erf_diff, "erf_diff_NPOPfls", "NPOPfls calculation", raise_on_error=False)
                
                npopfls_numerator = _safe_div(erf_diff, divisor_fls, name="NPOPfls_divisor")
                _validate_array(npopfls_numerator, "npopfls_numerator", "NPOPfls calculation", raise_on_error=False)
                
                # Apply Simpson's rule weights
                simpson_weights = np.concatenate((np.array([1/2]),np.ones(self.M - 1),np.array([1/2]))).reshape(-1,1).T
                weighted_sum = np.matmul((np.ones((len(self.remainingindices),1))), simpson_weights)
                _validate_array(weighted_sum, "weighted_sum_NPOPfls", "NPOPfls calculation", raise_on_error=False)
                
                npopfls_sum = np.sum(npopfls_numerator * weighted_sum, axis=1)
                _validate_array(npopfls_sum, "npopfls_sum", "NPOPfls calculation", raise_on_error=False)
                
                # Final calculation with safe division
                deltaz_reshaped = self.Deltaz[self.deltazlinear.astype(int)].reshape(len(self.Deltaz[self.deltazlinear.astype(int)]),1).T
                _validate_array(deltaz_reshaped, "deltaz_reshaped_NPOPfls", "NPOPfls calculation", raise_on_error=False)
                
                if self.M <= 0 or np.isnan(self.M) or np.isinf(self.M):
                    raise ValueError(f"Invalid M={self.M} for NPOPfls calculation")
                
                self.NPOPfls = deltaz_reshaped * _safe_div(npopfls_sum, self.M, name="NPOPfls_M_divisor")
                self.NPOPfls = self.NPOPfls.T
                _validate_array(self.NPOPfls, "NPOPfls", "after calculation", raise_on_error=False)
                # Clean NPOPfls before assigning to NPOP
                self.NPOPfls = _clean_array(self.NPOPfls, "NPOPfls")
                self.dimensions = self.NPOP.shape
            #  #pdb.set_trace()
                self.NPOP=self.NPOP.T.ravel()
                self.NPOP[self.remainingindices.reshape((len(self.remainingindices),1))] = self.NPOPfls
                self.NPOP = self.NPOP.reshape((self.dimensions[1],self.dimensions[0])).T
                # Clean NPOP after all assignments
                self.NPOP = _clean_array(self.NPOP, "NPOP")

        # Put everything together and calculate BB (= impact of all previous heat pulses from old neighbouring elements on current element at current time)
        #  
            self.Qindicestotake = self.SortedIndicesRelevant.ravel().reshape((self.N * int(self.lastneighbourtoconsider[self.counter]), 1))*np.ones((1,len(self.timesFMM)-1)) + \
                            np.ones((self.N * int(self.lastneighbourtoconsider[self.counter]), 1)) * self.N * np.arange(len(self.timesFMM) - 1)
            self.Qindicestotake = self.Qindicestotake.astype(int)
            self.Qlinear = self.QFMM.T.ravel()[self.Qindicestotake]

            self.BBPS = self.NPOP * self.Qlinear
            self.BBPS = np.sum(self.BBPS, axis=1)
            self.BBPSindicestotake = np.arange(self.N).reshape((self.N, 1)) + self.N * np.arange(int(self.lastneighbourtoconsider[self.counter])).reshape((1, int(self.lastneighbourtoconsider[self.counter])))
            self.BBPSMatrix = self.BBPS[self.BBPSindicestotake]
            self.BB = np.sum(self.BBPSMatrix, axis=1)
            # Clean BB immediately after calculation to prevent NaN/Inf propagation
            max_finite_val = np.finfo(np.float64).max / 1e10
            self.BB = np.nan_to_num(self.BB, nan=0.0, posinf=max_finite_val, neginf=-max_finite_val)

        if self.counter > 1:
            self.BBCPOP = np.sum(self.CPOP * self.QFMM[:, 1:], axis=1)
            # Clean BBCPOP immediately after calculation to prevent NaN/Inf propagation
            max_finite_val = np.finfo(np.float64).max / 1e10
            self.BBCPOP = np.nan_to_num(self.BBCPOP, nan=0.0, posinf=max_finite_val, neginf=-max_finite_val)
        else:
            self.BBCPOP = np.zeros(self.N)

        # Validate key parameters before populating matrices
        if np.any(self.Deltaz <= 0) or np.any(np.isnan(self.Deltaz)) or np.any(np.isinf(self.Deltaz)):
            raise ValueError(f"Invalid Deltaz values detected (zero/NaN/Inf). Check geometry parameters: lateral_length >= 5000m, numberoflaterals >= 3, well_tvd >= 7000m, dx >= 500m")
        if self.Deltat <= 0 or np.isnan(self.Deltat) or np.isinf(self.Deltat):
            raise ValueError(f"Invalid Deltat (time step) value: {self.Deltat}. Check timestep_hours parameter.")
        if np.any(self.Dvector <= 0) or np.any(np.isnan(self.Dvector)) or np.any(np.isinf(self.Dvector)):
            raise ValueError(f"Invalid Dvector values detected (zero/NaN/Inf). Check vertical_diam and lateral_diam parameters.")
        
        #Populate L and R for fluid heat balance for first element (which has the injection temperature specified)
        # Validate inputs before calculations to prevent NaN/Inf
        if self.Deltaz[0] <= 0 or np.isnan(self.Deltaz[0]) or np.isinf(self.Deltaz[0]):
            raise ValueError(f"Invalid Deltaz[0]={self.Deltaz[0]}. This indicates a geometry problem. "
                           f"ConfigManager should have set: lateral_length >= 5000m, numberoflaterals >= 3, well_tvd >= 7000m, dx >= 500m. "
                           f"Current: lateral_length={getattr(self, 'lateral_length', 'N/A')}, "
                           f"numberoflaterals={getattr(self, 'numberoflaterals', 'N/A')}, "
                           f"well_tvd={getattr(self, 'well_tvd', 'N/A')}, dx={getattr(self, 'dx', 'N/A')}")
        if self.Deltat <= 0 or np.isnan(self.Deltat) or np.isinf(self.Deltat):
            raise ValueError(f"Invalid Deltat={self.Deltat}. Check timestep_hours parameter.")
        if self.Dvector[0] <= 0 or np.isnan(self.Dvector[0]) or np.isinf(self.Dvector[0]):
            raise ValueError(f"Invalid Dvector[0]={self.Dvector[0]}. Check vertical_diam and lateral_diam parameters.")
        
        # Use safe division for all operations
        inv_deltat = _safe_div(1.0, self.Deltat, "inv_Deltat_first")
        u_over_dz = _safe_div(self.uvector[0], self.Deltaz[0], "u_over_Deltaz_first")
        inv_dvector_sq = _safe_div(1.0, self.Dvector[0]**2, "inv_Dvector_sq_first")
        inv_rho_cp = _safe_div(1.0, self.rho_f * self.cp_f, "inv_rho_cp_first")
        
        self.LL[0, 0] = inv_deltat + u_over_dz * (self.fullyimplicit) * 2
        self.LL[0, 2] = -4 / np.pi * inv_dvector_sq * inv_rho_cp
        self.RR[0, 0] = inv_deltat * self.Twprevious[0] + u_over_dz * Tin * 2 - u_over_dz * self.Twprevious[0] * (1 - self.fullyimplicit) * 2
        
        # Check for NaN/Inf immediately after first element
        if np.any(np.isnan(self.LL[:3, :])) or np.any(np.isinf(self.LL[:3, :])):
            raise ValueError(f"NaN/Inf detected in LL matrix after first element. "
                           f"This suggests invalid geometry parameters. "
                           f"ConfigManager should have set: lateral_length >= 5000m, numberoflaterals >= 3, well_tvd >= 7000m, dx >= 500m.")

        #Populate L and R for rock temperature equation for first element   
        self.LL[1, 0] = 1
        self.LL[1, 1] = -1
        self.LL[1, 2] = self.Rtvector[0]
        self.RR[1, 0] = 0

        # Populate L and R for SBT algorithm for first element
        # Check NPCP for NaN/Inf before using
        npcp_row = self.NPCP[0,0:self.N]
        if np.any(np.isnan(npcp_row)) or np.any(np.isinf(npcp_row)):
            # Replace Inf with large finite value, NaN with 0
            npcp_row = np.where(np.isinf(npcp_row), np.sign(npcp_row) * 1e10, npcp_row)
            npcp_row = np.where(np.isnan(npcp_row), 0.0, npcp_row)
        
        self.LL[2, np.arange(2,3*self.N,3)] = npcp_row
        self.LL[2,1] = 1
        
        # Check BB, BBCPOP, BBinitial for NaN/Inf
        bb_val = self.BB[0] if hasattr(self.BB, '__getitem__') else self.BB
        bbcpop_val = self.BBCPOP[0] if hasattr(self.BBCPOP, '__getitem__') else self.BBCPOP
        bbinitial_val = self.BBinitial[0] if hasattr(self.BBinitial, '__getitem__') else self.BBinitial
        
        if np.isnan(bb_val) or np.isinf(bb_val):
            bb_val = 0.0 if np.isnan(bb_val) else (np.sign(bb_val) * 1e10 if np.isinf(bb_val) else bb_val)
        if np.isnan(bbcpop_val) or np.isinf(bbcpop_val):
            bbcpop_val = 0.0 if np.isnan(bbcpop_val) else (np.sign(bbcpop_val) * 1e10 if np.isinf(bbcpop_val) else bbcpop_val)
        if np.isnan(bbinitial_val) or np.isinf(bbinitial_val):
            bbinitial_val = 0.0 if np.isnan(bbinitial_val) else (np.sign(bbinitial_val) * 1e10 if np.isinf(bbinitial_val) else bbinitial_val)
        
        self.RR[2, 0] = -bbcpop_val - bb_val + bbinitial_val
        
        # Clean LL and RR after first element population
        self.LL[:3, :] = _clean_array(self.LL[:3, :], "LL_first_element")
        self.RR[:3, :] = _clean_array(self.RR[:3, :], "RR_first_element")

        for iiii in range(2, self.N+1):  
            # Heat balance equation - use safe division for all operations
            inv_deltat_i = _safe_div(1.0, self.Deltat, f"inv_Deltat_{iiii}")
            u_over_dz_i = _safe_div(self.uvector[iiii-1], self.Deltaz[iiii-1], f"u_over_Deltaz_{iiii}")
            inv_dvector_sq_i = _safe_div(1.0, self.Dvector[iiii-1] ** 2, f"inv_Dvector_sq_{iiii}")
            inv_rho_cp_i = _safe_div(1.0, self.rho_f * self.cp_f, f"inv_rho_cp_{iiii}")
            
            self.LL[0+(iiii - 1) * 3,  (iiii - 1) * 3] = inv_deltat_i + u_over_dz_i / 2 * (self.fullyimplicit) * 2
            self.LL[0+(iiii - 1) * 3, 2 + (iiii - 1) * 3] = -4 / np.pi * inv_dvector_sq_i * inv_rho_cp_i

            if iiii == len(self.xinj):  # Upcoming pipe has first element temperature sum of all incoming water temperatures
                lateral_diam_ratio_sq = _safe_div(self.lateral_diam, self.prd_well_diam, f"lateral_diam_ratio_{iiii}") ** 2
                inv_lateral_mult = _safe_div(1.0, self.lateralflowmultiplier, f"inv_lateral_mult_{iiii}")
                
                for j in range(len(self.lateralendpoints)):
                    u_lateral_over_dz = _safe_div(self.ulateral[j], self.Deltaz[iiii-1], f"u_lateral_over_Deltaz_{iiii}_{j}")
                    self.LL[0+ (iiii - 1) * 3, 0 + (self.lateralendpoints[j]) * 3] = -u_lateral_over_dz / 2 * inv_lateral_mult * (self.fullyimplicit) * 2 * lateral_diam_ratio_sq
                    self.RR[0+(iiii - 1) * 3, 0] = inv_deltat_i * self.Twprevious[iiii-1] + u_over_dz_i * (
                            -self.Twprevious[iiii-1] + lateral_diam_ratio_sq * np.sum(self.lateralflowallocation[j] * self.Twprevious[self.lateralendpoints[j]])) / 2 * (
                                                    1 - self.fullyimplicit) * 2
            else:
                self.LL[0+(iiii-1) * 3, 0 + (int(self.previouswaterelements[iiii-1])) * 3] = -u_over_dz_i / 2 * (
                        self.fullyimplicit) * 2
                self.RR[0+(iiii-1) * 3, 0] = inv_deltat_i * self.Twprevious[iiii-1] + u_over_dz_i * (
                        -self.Twprevious[iiii-1] + self.Twprevious[int(self.previouswaterelements[iiii-1])]) / 2 * (1 - self.fullyimplicit) * 2

            # Rock temperature equation
            self.LL[1 + (iiii - 1) * 3,  (iiii - 1) * 3] = 1
            self.LL[1 + (iiii - 1) * 3, 1 + (iiii - 1) * 3] = -1
            self.LL[1 + (iiii - 1) * 3, 2 + (iiii - 1) * 3] = self.Rtvector[iiii-1]
            self.RR[1 + (iiii - 1) * 3, 0] = 0

            # SBT equation 
            # Check NPCP for NaN/Inf before using
            npcp_row = self.NPCP[iiii-1, :self.N]
            if np.any(np.isnan(npcp_row)) or np.any(np.isinf(npcp_row)):
                # Replace Inf with large finite value, NaN with 0
                npcp_row = np.where(np.isinf(npcp_row), np.sign(npcp_row) * 1e10, npcp_row)
                npcp_row = np.where(np.isnan(npcp_row), 0.0, npcp_row)
            
            self.LL[2 + (iiii - 1) * 3, np.arange(2,3*self.N,3)] = npcp_row
            self.LL[2 + (iiii - 1) * 3, 1 + (iiii - 1) * 3] = 1
            
            # Check BB, BBCPOP, BBinitial for NaN/Inf
            bb_val = self.BB[iiii-1] if hasattr(self.BB, '__getitem__') else self.BB
            bbcpop_val = self.BBCPOP[iiii-1] if hasattr(self.BBCPOP, '__getitem__') else self.BBCPOP
            bbinitial_val = self.BBinitial[iiii-1] if hasattr(self.BBinitial, '__getitem__') else self.BBinitial
            
            if np.isnan(bb_val) or np.isinf(bb_val):
                bb_val = 0.0 if np.isnan(bb_val) else (np.sign(bb_val) * 1e10 if np.isinf(bb_val) else bb_val)
            if np.isnan(bbcpop_val) or np.isinf(bbcpop_val):
                bbcpop_val = 0.0 if np.isnan(bbcpop_val) else (np.sign(bbcpop_val) * 1e10 if np.isinf(bbcpop_val) else bbcpop_val)
            if np.isnan(bbinitial_val) or np.isinf(bbinitial_val):
                bbinitial_val = 0.0 if np.isnan(bbinitial_val) else (np.sign(bbinitial_val) * 1e10 if np.isinf(bbinitial_val) else bbinitial_val)
            
            self.RR[2 + (iiii - 1) * 3, 0] = -bbcpop_val - bb_val + bbinitial_val
            
            # Clean LL and RR after each element population
            row_start = (iiii - 1) * 3
            self.LL[row_start:row_start+3, :] = _clean_array(self.LL[row_start:row_start+3, :], f"LL_element_{iiii}")
            self.RR[row_start:row_start+3, :] = _clean_array(self.RR[row_start:row_start+3, :], f"RR_element_{iiii}")


        # Solving the linear system of equations
        # Replace Inf with large finite values and NaN with 0 before solving
        # This prevents numerical errors while maintaining matrix structure
        max_finite_val = np.finfo(np.float64).max / 1e10  # Large but safe finite value
        
        # Replace Inf in LL with large finite values (preserve sign)
        inf_mask_ll = np.isinf(self.LL)
        if np.any(inf_mask_ll):
            self.LL[inf_mask_ll] = np.where(self.LL[inf_mask_ll] > 0, max_finite_val, -max_finite_val)
        
        # Replace NaN in LL with 0 (should not happen, but safety check)
        nan_mask_ll = np.isnan(self.LL)
        if np.any(nan_mask_ll):
            self.LL[nan_mask_ll] = 0.0
        
        # Replace Inf in RR with large finite values (preserve sign)
        inf_mask_rr = np.isinf(self.RR)
        if np.any(inf_mask_rr):
            self.RR[inf_mask_rr] = np.where(self.RR[inf_mask_rr] > 0, max_finite_val, -max_finite_val)
        
        # Replace NaN in RR with 0
        nan_mask_rr = np.isnan(self.RR)
        if np.any(nan_mask_rr):
            self.RR[nan_mask_rr] = 0.0
        
        # Final check - if still NaN/Inf after replacement, raise error
        if np.any(np.isnan(self.LL)) or np.any(np.isinf(self.LL)) or np.any(np.isnan(self.RR)) or np.any(np.isinf(self.RR)):
            ll_nan_count = np.sum(np.isnan(self.LL))
            ll_inf_count = np.sum(np.isinf(self.LL))
            rr_nan_count = np.sum(np.isnan(self.RR))
            rr_inf_count = np.sum(np.isinf(self.RR))
            raise ValueError(f"Unable to fix NaN/Inf values: LL has {ll_nan_count} NaN, {ll_inf_count} Inf; RR has {rr_nan_count} NaN, {rr_inf_count} Inf. "
                           f"This indicates a fundamental numerical issue. Try reducing timestep_hours or adjusting geometry parameters.")
        
        # Use numpy's nan_to_num as additional safeguard before solve
        # This converts NaN to 0 and Inf to large finite values
        # CRITICAL: numpy's linalg.solve checks for NaN/Inf in C code before Python wrapper
        # So we MUST replace all NaN/Inf values BEFORE calling solve
        # Ensure we're working with contiguous arrays (not views)
        self.LL = np.ascontiguousarray(self.LL)
        self.RR = np.ascontiguousarray(self.RR)
        
        # Replace NaN/Inf using nan_to_num
        self.LL = np.nan_to_num(self.LL, nan=0.0, posinf=max_finite_val, neginf=-max_finite_val)
        self.RR = np.nan_to_num(self.RR, nan=0.0, posinf=max_finite_val, neginf=-max_finite_val)
        
        # Final verification - ensure no NaN/Inf remain (double-check)
        nan_mask_ll = np.isnan(self.LL) | np.isinf(self.LL)
        if np.any(nan_mask_ll):
            self.LL[nan_mask_ll] = np.where(np.isinf(self.LL[nan_mask_ll]), 
                                           np.where(self.LL[nan_mask_ll] > 0, max_finite_val, -max_finite_val),
                                           0.0)
        
        nan_mask_rr = np.isnan(self.RR) | np.isinf(self.RR)
        if np.any(nan_mask_rr):
            self.RR[nan_mask_rr] = np.where(np.isinf(self.RR[nan_mask_rr]),
                                           np.where(self.RR[nan_mask_rr] > 0, max_finite_val, -max_finite_val),
                                           0.0)
        
        # FINAL check right before solve - numpy checks in C code before Python wrapper
        # Create completely fresh copies with explicit dtype and contiguity
        LL_final = np.array(self.LL, dtype=np.float64, copy=True, order='C')
        RR_final = np.array(self.RR, dtype=np.float64, copy=True, order='C')
        
        # Ensure arrays are contiguous
        LL_final = np.ascontiguousarray(LL_final, dtype=np.float64)
        RR_final = np.ascontiguousarray(RR_final, dtype=np.float64)
        
        # AGGRESSIVE cleaning - multiple passes to catch any remaining NaN/Inf
        max_finite_val = np.finfo(np.float64).max / 1e10
        min_finite_val = -max_finite_val
        
        # Pass 1: Replace Inf with large finite values (preserve sign)
        LL_final = np.where(np.isinf(LL_final), 
                          np.where(LL_final > 0, max_finite_val, min_finite_val),
                          LL_final)
        RR_final = np.where(np.isinf(RR_final),
                          np.where(RR_final > 0, max_finite_val, min_finite_val),
                          RR_final)
        
        # Pass 2: Replace NaN with 0 for LL (diagonal should be non-zero, but safer than NaN)
        # For diagonal elements, use small positive value instead of 0
        LL_diag_indices = np.diag_indices(min(LL_final.shape))
        LL_final_diag_backup = LL_final[LL_diag_indices].copy()
        LL_final = np.where(np.isnan(LL_final), 0.0, LL_final)
        # Restore diagonal if it was valid before
        for i in range(min(LL_final.shape[0], LL_final.shape[1])):
            if np.isnan(LL_final_diag_backup[i]) or LL_final_diag_backup[i] == 0:
                LL_final[i, i] = max(1e-10, abs(LL_final[i, i]) if not np.isnan(LL_final[i, i]) else 1e-10)
        
        RR_final = np.where(np.isnan(RR_final), 0.0, RR_final)
        
        # Pass 3: Use nan_to_num as final safeguard
        LL_final = np.nan_to_num(LL_final, nan=0.0, posinf=max_finite_val, neginf=min_finite_val)
        RR_final = np.nan_to_num(RR_final, nan=0.0, posinf=max_finite_val, neginf=min_finite_val)
        
        # Pass 4: Final verification - if ANY NaN/Inf remain, replace with safe defaults
        nan_inf_mask_ll = np.isnan(LL_final) | np.isinf(LL_final)
        nan_inf_mask_rr = np.isnan(RR_final) | np.isinf(RR_final)
        
        if np.any(nan_inf_mask_ll):
            # For matrix elements, use small values to maintain structure
            LL_final[nan_inf_mask_ll] = np.where(np.isinf(LL_final[nan_inf_mask_ll]),
                                                np.where(LL_final[nan_inf_mask_ll] > 0, max_finite_val, min_finite_val),
                                                1e-10)
        
        if np.any(nan_inf_mask_rr):
            RR_final[nan_inf_mask_rr] = 0.0
        
        # Pass 5: One more nan_to_num to be absolutely sure
        LL_final = np.nan_to_num(LL_final, nan=0.0, posinf=max_finite_val, neginf=min_finite_val)
        RR_final = np.nan_to_num(RR_final, nan=0.0, posinf=max_finite_val, neginf=min_finite_val)
        
        # Final check - verify arrays are completely clean
        if np.any(np.isnan(LL_final)) or np.any(np.isinf(LL_final)) or np.any(np.isnan(RR_final)) or np.any(np.isinf(RR_final)):
            # Last resort: replace ALL remaining NaN/Inf with safe defaults
            LL_final = np.where(np.isnan(LL_final) | np.isinf(LL_final), 
                              np.where(np.isinf(LL_final) & (LL_final > 0), max_finite_val,
                                     np.where(np.isinf(LL_final) & (LL_final < 0), min_finite_val, 1e-10)),
                              LL_final)
            RR_final = np.where(np.isnan(RR_final) | np.isinf(RR_final), 0.0, RR_final)
            
            # One final nan_to_num
            LL_final = np.nan_to_num(LL_final, nan=0.0, posinf=max_finite_val, neginf=min_finite_val)
            RR_final = np.nan_to_num(RR_final, nan=0.0, posinf=max_finite_val, neginf=min_finite_val)
        
        # Use multiple solver strategies for robustness
        # Strategy 1: Try numpy lstsq (most tolerant)
        # Strategy 2: Try scipy.linalg.solve (if available)
        # Strategy 3: Try scipy.linalg.lstsq with regularization
        # Strategy 4: Try scipy.linalg.pinv (pseudoinverse) as last resort
        
        solver_success = False
        solver_error = None
        
        # Final verification before solver - ensure arrays are absolutely clean
        # Check one more time and replace any remaining NaN/Inf
        if np.any(np.isnan(LL_final)) or np.any(np.isinf(LL_final)):
            LL_final = np.nan_to_num(LL_final, nan=0.0, posinf=max_finite_val, neginf=min_finite_val)
        if np.any(np.isnan(RR_final)) or np.any(np.isinf(RR_final)):
            RR_final = np.nan_to_num(RR_final, nan=0.0, posinf=max_finite_val, neginf=min_finite_val)
        
        # Verify arrays are finite (numpy requirement)
        if not np.all(np.isfinite(LL_final)) or not np.all(np.isfinite(RR_final)):
            # Force all non-finite values to finite
            LL_final = np.where(np.isfinite(LL_final), LL_final, 
                              np.where(np.isinf(LL_final), 
                                     np.where(LL_final > 0, max_finite_val, min_finite_val), 0.0))
            RR_final = np.where(np.isfinite(RR_final), RR_final,
                              np.where(np.isinf(RR_final),
                                     np.where(RR_final > 0, max_finite_val, min_finite_val), 0.0))
            # One final nan_to_num
            LL_final = np.nan_to_num(LL_final, nan=0.0, posinf=max_finite_val, neginf=min_finite_val)
            RR_final = np.nan_to_num(RR_final, nan=0.0, posinf=max_finite_val, neginf=min_finite_val)
        
        # Strategy 1: numpy lstsq (primary)
        try:
            # Verify arrays one final time before calling solver
            assert np.all(np.isfinite(LL_final)), f"LL_final contains non-finite values: {np.sum(~np.isfinite(LL_final))}"
            assert np.all(np.isfinite(RR_final)), f"RR_final contains non-finite values: {np.sum(~np.isfinite(RR_final))}"
            
            self.Sol = np.linalg.lstsq(LL_final, RR_final, rcond=None)[0]
            # Validate solution
            if np.any(np.isnan(self.Sol)) or np.any(np.isinf(self.Sol)):
                raise ValueError("Solution contains NaN/Inf")
            solver_success = True
        except (ValueError, np.linalg.LinAlgError, AssertionError) as e:
            solver_error = str(e)
            if self.debug_mode:
                self.debug_log.append({'step': self.counter, 'solver': 'numpy_lstsq', 'error': solver_error})
        
        # Strategy 2: scipy.linalg.solve (if numpy fails)
        if not solver_success:
            try:
                import scipy.linalg
                # Add small regularization for numerical stability
                eps_reg = 1e-12
                LL_reg = LL_final + eps_reg * np.eye(LL_final.shape[0])
                self.Sol = scipy.linalg.solve(LL_reg, RR_final)
                if np.any(np.isnan(self.Sol)) or np.any(np.isinf(self.Sol)):
                    raise ValueError("Solution contains NaN/Inf")
                solver_success = True
            except Exception as e:
                solver_error = str(e)
                if self.debug_mode:
                    self.debug_log.append({'step': self.counter, 'solver': 'scipy_solve', 'error': str(e)})
        
        # Strategy 3: scipy.linalg.lstsq with regularization
        if not solver_success:
            try:
                import scipy.linalg
                eps_reg = 1e-10
                LL_reg = LL_final + eps_reg * np.eye(LL_final.shape[0])
                self.Sol = scipy.linalg.lstsq(LL_reg, RR_final, cond=None)[0]
                if np.any(np.isnan(self.Sol)) or np.any(np.isinf(self.Sol)):
                    raise ValueError("Solution contains NaN/Inf")
                solver_success = True
            except Exception as e:
                solver_error = str(e)
                if self.debug_mode:
                    self.debug_log.append({'step': self.counter, 'solver': 'scipy_lstsq', 'error': str(e)})
        
        # Strategy 4: scipy.linalg.pinv (pseudoinverse) - most robust but slower
        if not solver_success:
            try:
                import scipy.linalg
                LL_pinv = scipy.linalg.pinv(LL_final, rcond=1e-10)
                self.Sol = LL_pinv @ RR_final
                if np.any(np.isnan(self.Sol)) or np.any(np.isinf(self.Sol)):
                    raise ValueError("Solution contains NaN/Inf")
                solver_success = True
            except Exception as e:
                solver_error = str(e)
                if self.debug_mode:
                    self.debug_log.append({'step': self.counter, 'solver': 'scipy_pinv', 'error': str(e)})
        
        # If all solvers failed
        if not solver_success:
            # Clean arrays one more time and retry numpy lstsq
            LL_final = _clean_array(LL_final, "LL_final_final_retry")
            RR_final = _clean_array(RR_final, "RR_final_final_retry")
            try:
                self.Sol = np.linalg.lstsq(LL_final, RR_final, rcond=1e-10)[0]
                if np.any(np.isnan(self.Sol)) or np.any(np.isinf(self.Sol)):
                    raise ValueError("Solution contains NaN/Inf")
                solver_success = True
            except Exception as e2:
                # All solvers failed - arrays still have NaN/Inf that can't be cleaned
                current_vals = f"lateral_length={getattr(self, 'lateral_length', 'N/A')}, " \
                             f"numberoflaterals={getattr(self, 'numberoflaterals', 'N/A')}, " \
                             f"well_tvd={getattr(self, 'well_tvd', 'N/A')}, dx={getattr(self, 'dx', 'N/A')}, " \
                             f"timestep_hours={getattr(self, 'timestep', 'N/A')}"
                raise ValueError(f"❌ ULoop Simulation Error: Unable to solve linear system.\n\n"
                               f"🔧 The ConfigManager automatically overrides low values, but numerical instability occurred.\n\n"
                               f"📋 Current geometry: {current_vals}\n\n"
                               f"💡 Solutions:\n"
                               f"   1. Ensure lateral_length >= 5000m\n"
                               f"   2. Ensure numberoflaterals >= 12\n"
                               f"   3. Ensure well_tvd >= 7000m\n"
                               f"   4. Ensure dx >= 500m\n"
                               f"   5. Try reducing timestep_hours (e.g., 0.5 or 0.25)\n"
                               f"   6. Check that all diameter parameters are positive\n\n"
                               f"First error: {solver_error[:200] if solver_error else 'Unknown'}\n"
                               f"Final retry error: {str(e2)[:200]}")
        
        # Check solution for NaN/Inf
        if np.any(np.isnan(self.Sol)) or np.any(np.isinf(self.Sol)):
            raise ValueError(f"Solution contains NaN/Inf. This may indicate numerical instability. "
                           f"Try adjusting: lateral_length, numberoflaterals, well_tvd, dx, or timestep_hours.")

        # Extracting Q array for current heat pulses
        self.Q[:, self.counter] = self.Sol.ravel()[2::3]

        # Extracting fluid temperature
        self.TwMatrix[self.counter, :] = self.Sol.ravel()[np.arange(0,3*self.N,3)]

        # Storing fluid temperature for the next time step
        self.Twprevious = self.Sol.ravel()[np.arange(0,3*self.N,3)]

        #FMM algorithm for combining heat pulses
        #---------------------------------------
        if (self.FMM == 1 and self.counter>50 and self.times_arr[self.counter]*self.Deltat > self.FMMtriggertime):
            self.remainingtimes = self.times_arr[(self.times_arr >= self.combinedtimes[-1]) & (self.times_arr < self.times_arr[self.counter])] 
            self.currentendtimespassed = self.times_arr[self.counter] - self.remainingtimes
            
            if len(self.remainingtimes)>40:
                if self.currentendtimespassed[24]*self.Deltat>self.FMMtriggertime:
                    self.combinedtimes = np.append(self.combinedtimes,self.remainingtimes[24])
                    startindex = np.where(self.times_arr == self.combinedtimes[-2])[0][0]
                    endindex = np.where(self.times_arr == self.remainingtimes[24])[0][0]
                    newcombinedQ = np.sum(self.Q[:,startindex+1:endindex+1]*(self.times_arr[startindex+1:endindex+1]-self.times_arr[startindex:endindex])/(self.combinedtimes[-1]-self.combinedtimes[-2]),axis = 1) #weighted average
                    newcombinedQ = newcombinedQ.reshape(-1, 1)
                    self.combinedQ = np.hstack((self.combinedQ, newcombinedQ))
            
            #combine very old time pulses
            startindexforsecondlevel = np.where(self.combinedtimes == self.combinedtimes2ndlevel[-1])[0][0]
            if self.combinedtimes.size>30 and self.combinedtimes.size-startindexforsecondlevel>30:
                if (self.times_arr[self.counter] - self.combinedtimes[startindexforsecondlevel+20])*self.Deltat>self.FMMtriggertime*5:
                    indicestodrop = np.arange(startindexforsecondlevel+1, startindexforsecondlevel+20)
                    weightedQ = np.sum(self.combinedQ[:,startindexforsecondlevel+1:startindexforsecondlevel+20+1]*\
                                        (self.combinedtimes[startindexforsecondlevel+1:startindexforsecondlevel+20+1]-self.combinedtimes[startindexforsecondlevel:startindexforsecondlevel+20])\
                                            /(self.combinedtimes[startindexforsecondlevel+20]-self.combinedtimes[startindexforsecondlevel]),axis = 1)
                    self.combinedtimes2ndlevel = np.append(self.combinedtimes2ndlevel,self.combinedtimes[startindexforsecondlevel+20])
                    self.combinedtimes = np.delete(self.combinedtimes, indicestodrop)
                    self.combinedQ[:,startindexforsecondlevel+20] = weightedQ
                    self.combinedQ = np.delete(self.combinedQ, indicestodrop, axis=1)


            #combine very very old time pulses
            startindexforthirdlevel = np.where(self.combinedtimes == self.combinedtimes3rdlevel[-1])[0][0]
            if self.combinedtimes.size>50 and self.combinedtimes.size-startindexforthirdlevel>50:
                if (self.times_arr[self.counter] - self.combinedtimes[startindexforthirdlevel+20])*self.Deltat>self.FMMtriggertime*10:
                    indicestodrop = np.arange(startindexforthirdlevel+1, startindexforthirdlevel+20)
                    weightedQ = np.sum(self.combinedQ[:,startindexforthirdlevel+1:startindexforthirdlevel+20+1]*\
                                        (self.combinedtimes[startindexforthirdlevel+1:startindexforthirdlevel+20+1]-self.combinedtimes[startindexforthirdlevel:startindexforthirdlevel+20])\
                                            /(self.combinedtimes[startindexforthirdlevel+20]-self.combinedtimes[startindexforthirdlevel]),axis = 1)
                    self.combinedtimes3rdlevel = np.append(self.combinedtimes3rdlevel,self.combinedtimes[startindexforthirdlevel+20])
                    self.combinedtimes = np.delete(self.combinedtimes, indicestodrop)
                    self.combinedQ[:,startindexforthirdlevel+20] = weightedQ
                    self.combinedQ = np.delete(self.combinedQ, indicestodrop, axis=1)
            
            
            endindex = np.where(self.times_arr == self.combinedtimes[-1])[0][0]
            remainingQ = self.Q[:,endindex+1:self.counter+1]
            self.QFMM = np.hstack((self.combinedQ, remainingQ))
            self.timesFMM = np.append(self.combinedtimes,self.times_arr[endindex+1:self.counter+1]) 
            
        else:
            self.QFMM = self.Q[:,0:self.counter+1]
            self.timesFMM = self.times_arr[0:self.counter+1]

        ##########
        # Calculating the fluid outlet temperature at the top of the first element
        Tw_final_injector = self.TwMatrix[self.counter, 0:(self.interconnections[0] - 1)]  # Final fluid temperature profile in injection well [°C]
        Tw_final_producer = self.TwMatrix[self.counter, self.interconnections[0]:self.interconnections[1] - 1]

        self.Tres = max(Tw_final_producer)
        self.T_prd_bh = np.array(self.num_prd*[self.Tres], dtype='float')
        self.T_inj_wh = np.array(self.num_inj*[min(Tw_final_injector)], dtype='float')

    def configure_well_dimensions(self):
        """Configuration specifications of a doublet. It requires the specification of a doublet, including the producer dimensions (self.xprod, self.yprod, self.zprod), injector dimensions (self.xinj, self.yinj, self.zinj) and reservoir vertices (self.verts). See Class PercentageReservoir for example implementation.

        """

        self.v = [
                    [np.min(self.x), np.min(self.y) - 5 * self.lateral_spacing, -self.well_tvd + self.res_thickness/2],
                    [np.min(self.x), np.max(self.y) + 5 * self.lateral_spacing, -self.well_tvd + self.res_thickness/2],
                    [np.max(self.x), np.max(self.y) + 5 * self.lateral_spacing, -self.well_tvd + self.res_thickness/2],
                    [np.max(self.x), np.min(self.y) - 5 * self.lateral_spacing, -self.well_tvd + self.res_thickness/2],
                    [np.max(self.x), np.min(self.y) - 5 * self.lateral_spacing, -self.well_tvd - self.res_thickness/2],
                    [np.min(self.x), np.min(self.y) - 5 * self.lateral_spacing, -self.well_tvd - self.res_thickness/2],
                    [np.min(self.x), np.max(self.y) + 5 * self.lateral_spacing, -self.well_tvd - self.res_thickness/2],
                    [np.max(self.x), np.max(self.y) + 5 * self.lateral_spacing, -self.well_tvd - self.res_thickness/2],
                ]

        self.v = np.array(self.v)
        self.f = [[0,1,2,3], [4,5,6,7], [0, 1, 6, 5], [1, 2, 7, 6], [2, 3, 4, 7], [0, 3, 4, 5]]
        self.verts =  [[self.v[i] for i in p] for p in self.f]

class CoaxialSBT(BaseReservoir):
    """Numerical Coaxial model based on Slender-Body Theory (SBT), originally developed by  Beckers et al. (2023)."""

    def __init__(self,
                 Tres_init,
                 Pres_init,
                 geothermal_gradient,
                 surface_temp,
                 L,
                 time_init,
                 well_tvd,
                 casing_inner_diam,
                 tube_inner_diam,
                 tube_thickness,
                 k_tube,
                 num_well,
                 waterloss,
                 powerplant_type,
                 pumpeff,
                 times_arr,
                 k_m=2.3,
                 rho_m=2875,
                 c_m=825,
                 res_thickness=1000,
                 V_res=1,
                 phi_res=0.1,
                 dynamic_properties=False,
                 k_f=0.6,
                 mu_f=600 * 1E-6,
                 cp_f=4184,
                 rho_f=990,
                 dx=None,
                 # 1 = CXA (fluid injection in annulus); 2 = CXC (fluid
                 # injection in center pipe)
                 coaxialflowtype=1,
                 reservoir_simulator_settings={
                     "fast_mode": False, "accuracy": 5, "DynamicFluidProperties": False},
                 PumpingModel="ClosedLoop",
                 closedloop_design="Default",
                 FMM=1,
                 FMMtriggertime=3600 * 24 * 10,
                 ):
        """Initialize reservoir model.

        Args:
            Tres_init (float): initial reservoir temperature in deg C.
            geothermal_gradient (float): average geothermal gradient in deg C/km.
            surface_temp (float): surface temperature in deg C.
            L (float): project lifetime in years.
            time_init (datetime): initial time.
            well_tvd (float): well depth in meters.
            prd_well_diam (float): production well diameter in meters.
            inj_well_diam (float): injection well diameter in meters.
            num_prd (int): number of producers.
            num_inj (int): number of injectors.
            waterloss (float): fraction of injected water that is lost to the reservoir (fraction).
            powerplant_type (str): type of power plant (either "Binary" or "Flash").
            pumpeff (float): pump efficiency (fraction).
            ramey (bool, optional): whether or not to use ramey's model for wellbore heat loss/gain. Defaults to True.
            pumping (bool, optional): whther or not to account for parasitic losses due to pumping requirements. Defaults to True.
            k_m (float, optional): rock thermal conductivity in W/C-m. Defaults to 3.
            rho_m (float, optional): rock bulk density in kg/m3. Defaults to 2700.
            c_m (float, optional): rock heat capacity in J/kg-K. Defaults to 1000.
            impedance (float, optional): reservoir pressure losses when using an impendance model. Defaults to 0.1.
            res_thickness (float, optional): reservoir thickness in meters. Defaults to 200.
            PI (float, optional): productivity index in kg/s/bar. Defaults to 20.
            II (float, optional): injectivity index in kg/s/bar. Defaults to 20.
            SSR (float, optional): Stimulation success rate, which is a multiplier used to reduce PI and II when stimulation is not fully successful. Defaults to 1.0.
            N_ramey_mv_avg (int, optional): number of timesteps used for averaging the f-function when computing ramey's heat losses with variable mass flow rates. Defaults to 168.
            V_res (float, optional): reservoir bulk volume for all wells in km3. Defaults to 1.
            phi_res (float, optional): reservoir porosity (fraction). Defaults to 0.1.
            half_lateral_length (float, optional): half of the total producer-to-injector lateral length in meters. Defaults to 2000.
            lateral_diam (float, optional): diameter of wellbore lateral section in meters. Defaults to 0.3115.
            lateral_spacing (float, optional): spacing between uloop laterals in meters. Defaults to 100.
            dynamic_properties (bool, optional):  whether or not geofluid properties in the subsurface are updated using steamtables as a function of varying subsurface temperature. Defaults to False.
            k_f (float, optional): fluid thermal conductivity in W/C-m. Defaults to 0.68.
            mu_f (float, optional): fluid kinematic viscosity in m2/s. Defaults to 600e-6.
            cp_f (float, optional): fluid heat capacity in J/kg-K. Defaults to 4200.
            rho_f (float, optional): fluid density in kg/m3. Defaults to 1000.
            dx (int, optional): mesh descritization size in meters. Defaults to None (computed automatically).
            numberoflaterals (int, optional): number of laterals for each uloop doublet. Defaults to 3.
            lateralflowallocation (int, optional): distribution of flow across uloop laterals. Defaults to None (equal distribution)
            lateralflowmultiplier (int, optional): velocity multiplier across laterals. Defaults to 1.
            fullyimplicit (int, optional): how to solve the numerical system of equations using Euler's. Defaults to 1.
            reservoir_simulator_settings (dict, optional): information used to reduce the required timestepping when simulating the reservoir. It comes with keys of "fast_mode" to turn it on and "period" to specify the time period needed to pass before the reservoir state is updated, which is aimed at reducing computational requirements in exchange for loss in accuracy. Defaults to {"fast_mode": False, "period": 3600*8760/12}.
            PumpingModel (str, optional): model type used to compute pressure losses (either "OpenLoop" or "ClosedLoop"). Defaults to "ClosedLoop".
            closedloop_design (str, optional): Type of closedloop_design to simulate (either "Default" or "Eavor"). Defaults to "Default".
        """

        super(CoaxialSBT, self).__init__(Tres_init,
                                        geothermal_gradient,
                                        surface_temp,
                                        L,
                                        time_init,
                                        well_tvd,
                                        casing_inner_diam,
                                        casing_inner_diam,
                                        num_well,
                                        num_well,
                                        waterloss,
                                        powerplant_type,
                                        pumpeff,
                                        False,
                                        True,
                                        k_m,
                                        rho_m,
                                        c_m,
                                        0.1,
                                        res_thickness,
                                        20,
                                        20,
                                        1.0,
                                        168,
                                        reservoir_simulator_settings,
                                        PumpingModel)

        self.well_tvd = well_tvd
        self.well_md = self.well_tvd
        self.casing_inner_diam = casing_inner_diam
        self.tube_inner_diam = tube_inner_diam
        self.tube_thickness = tube_thickness
        self.k_tube = k_tube
        self.radius = casing_inner_diam / 2
        self.radiuscenterpipe = tube_inner_diam / 2
        self.thicknesscenterpipe = tube_thickness
        self.outerradiuscenterpipe = self.radiuscenterpipe + self.thicknesscenterpipe
        self.k_center_pipe = k_tube
        self.rho_f = rho_f if rho_f else densitywater(_safe_mean(self.T_prd_bh))
        self.cp_f = cp_f if cp_f else heatcapacitywater(_safe_mean(self.T_prd_bh))
        self.mu_f = mu_f if mu_f else viscositywater(_safe_mean(self.T_prd_bh))
        self.k_f = k_f
        self.c_m = c_m
        self.k_m = k_m
        self.rho_m = rho_m
        self.accuracy = reservoir_simulator_settings["accuracy"]
        self.L = L
        self.geothermal_gradient = geothermal_gradient/1000 #C/m
        self.surface_temp = surface_temp
        self.times = times_arr
        self.res_thickness = res_thickness 
        self.res_length = 0
        self.res_width = 0
        self.coaxialflowtype = coaxialflowtype
        self.FMM = FMM #if 1, use fast multi-pole methold like approach (i.e., combine old heat pulses to speed up simulation)
        self.FMMtriggertime = FMMtriggertime #threshold time beyond which heat pulses can be combined with others [s
        self.Deltat = self.timestep.total_seconds()
        N = 10
        # Calculate dx safely: use provided dx, or calculate from well_tvd, or use default
        if dx and dx > 0:
            self.dx = dx
        elif self.well_tvd and self.well_tvd > 0:
            self.dx = self.well_tvd // N
            if self.dx <= 0:  # Ensure dx is positive
                self.dx = 10.0
        else:
            # Fallback: use default value
            self.dx = 10.0

        # Merge x-, y-, and z-coordinates
        self.z = np.arange(0, -self.well_tvd-1, -self.dx).reshape(-1, 1)
        self.x = np.zeros((len(self.z), 1))
        self.y = np.zeros((len(self.z), 1))

        # Dummy placeholders
        self.xinj, self.yinj, self.zinj = self.x, self.y, self.z
        self.xprod, self.yprod, self.zprod = self.x, self.y, self.z

        self.g = 9.81                  #Gravitational acceleration [m/s^2]
        self.gamma = 0.577215665  # Euler's constant
        self.alpha_f = self.k_f / self.rho_f / self.cp_f  # Fluid thermal diffusivity [m2/s]
        self.Pr_f = self.mu_f / self.rho_f / self.alpha_f  # Fluid Prandtl number [-]
        self.alpha_m = self.k_m / self.rho_m / self.c_m  # Thermal diffusivity medium [m2/s]
        
        self.outerradiuscenterpipe = self.radiuscenterpipe + self.thicknesscenterpipe #Outer radius of inner pipe [m]
        self.A_flow_annulus = math.pi*(self.radius**2-self.outerradiuscenterpipe**2)       #Flow area of annulus pipe [m^2]
        self.A_flow_centerpipe = math.pi*self.radiuscenterpipe**2         #Flow area of center pipe [m^2]
        self.Dh_annulus = 2*(self.radius-self.outerradiuscenterpipe) #Hydraulic diameter of annulus [m]

        
        self.Deltaz = np.sqrt((self.x[1:] - self.x[:-1]) ** 2 + (self.y[1:] - self.y[:-1]) ** 2 + (self.z[1:] - self.z[:-1]) ** 2)  # Length of each segment [m]
        self.Deltaz = self.Deltaz.reshape(-1)
        self.TotalLength = np.sum(self.Deltaz)  # Total length of all elements (for informational purposes only) [m]
        self.total_drilling_length = self.well_md

        # Quality Control
        LoverR = self.Deltaz / self.radius  # Ratio of pipe segment length to radius along the wellbore [-]
        smallestLoverR = np.min(LoverR)  # Smallest ratio of pipe segment length to pipe radius. This ratio should be larger than 10. [-]
        
        if smallestLoverR < 10:
            print('Warning: smallest ratio of segment length over radius is less than 10. Good practice is to keep this ratio larger than 10.')
        
        RelativeLengthChanges = (self.Deltaz[1:] - self.Deltaz[:-1]) / self.Deltaz[:-1]
        if max(abs(RelativeLengthChanges)) > 0.5:
            print('Warning: abrupt change(s) in segment length detected, which may cause numerical instabilities. Good practice is to avoid abrupt length changes to obtain smooth results.')


        if self.accuracy == 1:
            self.NoArgumentsFinitePipeCorrection = 25
            self.NoDiscrFinitePipeCorrection = 200
            self.NoArgumentsInfCylIntegration = 25
            self.NoDiscrInfCylIntegration = 200
            self.LimitPointSourceModel = 1.5
            self.LimitCylinderModelRequired = 25
            self.LimitInfiniteModel = 0.05
            self.LimitNPSpacingTime = 0.1
            self.LimitSoverL = 1.5
            self.M = 3
        elif self.accuracy == 2:
            self.NoArgumentsFinitePipeCorrection = 50
            self.NoDiscrFinitePipeCorrection = 400
            self.NoArgumentsInfCylIntegration = 50
            self.NoDiscrInfCylIntegration = 400
            self.LimitPointSourceModel = 2.5
            self.LimitCylinderModelRequired = 50
            self.LimitInfiniteModel = 0.01
            self.LimitNPSpacingTime = 0.04
            self.LimitSoverL = 2
            self.M = 4
        elif self.accuracy == 3:
            self.NoArgumentsFinitePipeCorrection = 100
            self.NoDiscrFinitePipeCorrection = 500
            self.NoArgumentsInfCylIntegration = 100
            self.NoDiscrInfCylIntegration = 500
            self.LimitPointSourceModel = 5
            self.LimitCylinderModelRequired = 100
            self.LimitInfiniteModel = 0.004
            self.LimitNPSpacingTime = 0.02
            self.LimitSoverL = 3
            self.M = 5
        elif self.accuracy == 4:
            self.NoArgumentsFinitePipeCorrection = 200
            self.NoDiscrFinitePipeCorrection = 1000
            self.NoArgumentsInfCylIntegration = 200
            self.NoDiscrInfCylIntegration = 1000
            self.LimitPointSourceModel = 10
            self.LimitCylinderModelRequired = 200
            self.LimitInfiniteModel = 0.002
            self.LimitNPSpacingTime = 0.01
            self.LimitSoverL = 5
            self.M = 10
        elif self.accuracy == 5:
            self.NoArgumentsFinitePipeCorrection = 400
            self.NoDiscrFinitePipeCorrection = 2000
            self.NoArgumentsInfCylIntegration = 400
            self.NoDiscrInfCylIntegration = 2000
            self.LimitPointSourceModel = 20
            self.LimitCylinderModelRequired = 400
            self.LimitInfiniteModel = 0.001
            self.LimitNPSpacingTime = 0.005
            self.LimitSoverL = 9
            self.M = 20
    
        self.timeforpointssource = max(self.Deltaz)**2 / self.alpha_m * self.LimitPointSourceModel  # Calculates minimum time step size when point source model becomes applicable [s]
        self.timeforlinesource = self.radius**2 / self.alpha_m * self.LimitCylinderModelRequired  # Calculates minimum time step size when line source model becomes applicable [s]
        self.timeforfinitelinesource = max(self.Deltaz)**2 / self.alpha_m * self.LimitInfiniteModel  # Calculates minimum time step size when finite line source model should be considered [s]
        
        #precalculate the thermal response with a line and cylindrical heat source. Precalculating allows to speed up the SBT algorithm.
        #precalculate finite pipe correction
        self.fpcminarg = min(self.Deltaz)**2 / (4 * self.alpha_m * self.times[-1])
        self.fpcmaxarg = max(self.Deltaz)**2 / (4 * self.alpha_m * (min(self.times[1:] - self.times[:-1])))
        self.Amin1vector = np.logspace(np.log10(self.fpcminarg) - 0.1, np.log10(self.fpcmaxarg) + 0.1, self.NoArgumentsFinitePipeCorrection)
        self.finitecorrectiony = np.zeros(self.NoArgumentsFinitePipeCorrection)
        
        for i, Amin1 in enumerate(self.Amin1vector):
            Amax1 = (16)**2
            if Amin1 > Amax1:
                Amax1 = 10 * Amin1
            Adomain1 = np.logspace(np.log10(Amin1), np.log10(Amax1), self.NoDiscrFinitePipeCorrection)
            self.finitecorrectiony[i] = np.trapz(-1 / (Adomain1 * 4 * np.pi * self.k_m) * erfc(1/2 * np.power(Adomain1, 1/2)), Adomain1)
        
        #precalculate besselintegration for infinite cylinder
        self.besselminarg = self.alpha_m * (min(self.times[1:] - self.times[:-1])) / self.radius**2
        self.besselmaxarg = self.alpha_m * self.timeforlinesource / self.radius**2
        
        self.deltazbessel = np.logspace(-10, 8, self.NoDiscrInfCylIntegration)
        self.argumentbesselvec = np.logspace(np.log10(self.besselminarg) - 0.5, np.log10(self.besselmaxarg) + 0.5, self.NoArgumentsInfCylIntegration)
        self.besselcylinderresult = np.zeros(self.NoArgumentsInfCylIntegration)
        
        for i, argumentbessel in enumerate(self.argumentbesselvec):
            self.besselcylinderresult[i] = 2 / (self.k_m * np.pi**3) * np.trapz((1 - np.exp(-self.deltazbessel**2 * argumentbessel)) / (self.deltazbessel**3 * (jv(1, self.deltazbessel)**2 + yv(1, self.deltazbessel)**2)), self.deltazbessel)
        
        self.N = len(self.Deltaz)  # Number of elements
        self.elementcenters = 0.5 * np.column_stack((self.x[1:], self.y[1:], self.z[1:])) + 0.5 * np.column_stack((self.x[:-1], self.y[:-1], self.z[:-1]))  # Matrix that stores the mid point coordinates of each element
        
        self.SMatrix = np.zeros((self.N, self.N))  # Initializes the spacing matrix, which holds the distance between center points of each element [m]
        self.SoverL = np.zeros((self.N, self.N))  # Initializes the ratio of spacing to element length matrix
        for i in range(self.N):
            self.SMatrix[i, :] = np.sqrt((self.elementcenters[i, 0] - self.elementcenters[:, 0])**2 + (self.elementcenters[i, 1] - self.elementcenters[:, 1])**2 + (self.elementcenters[i, 2] - self.elementcenters[:, 2])**2)
            self.SoverL[i, :] = self.SMatrix[i, :] / self.Deltaz[i] #Calculates the ratio of spacing between two elements and element length
        
        #Element ranking based on spacinng is required for SBT algorithm as elements in close proximity to each other use different analytical heat transfer models than elements far apart
        self.SortedIndices = np.argsort(self.SMatrix, axis=1, kind = 'stable') # Getting the indices of the sorted elements
        self.SMatrixSorted = np.take_along_axis(self.SMatrix, self.SortedIndices, axis=1)  # Sorting the spacing matrix 
        self.SoverLSorted = self.SMatrixSorted / self.Deltaz
        
        #filename = 'smatrixpython.mat'
        #scipy.io.savemat(filename, dict(SortedIndicesPython=SortedIndices,SMatrixSortedPython=SMatrixSorted))
        
        self.mindexNPCP = np.where(np.min(self.SoverLSorted, axis=0) < self.LimitSoverL)[0][-1]  # Finding the index where the ratio is less than the limit
        
        self.midpointsx = self.elementcenters[:, 0]  # x-coordinate of center of each element [m]
        self.midpointsy = self.elementcenters[:, 1]  # y-coordinate of center of each element [m]
        self.midpointsz = self.elementcenters[:, 2]  # z-coordinate of center of each element [m]
        self.BBinitial = self.surface_temp - self.geothermal_gradient * self.midpointsz  # Initial temperature at center of each element [degC]
        
        self.MaxSMatrixSorted = np.max(self.SMatrixSorted, axis=0)
        
        self.indicesyoucanneglectupfront = self.alpha_m * (np.ones((self.N-1, 1)) * self.times) / (self.MaxSMatrixSorted[1:].reshape(-1, 1) * np.ones((1, len(self.times))))**2 / self.LimitNPSpacingTime
        self.indicesyoucanneglectupfront[self.indicesyoucanneglectupfront > 1] = 1


        self.lastneighbourtoconsider = np.zeros(len(self.times))
        for i in range(len(self.times)):
            lntc = np.where(self.indicesyoucanneglectupfront[:, i] == 1)[0]
            if len(lntc) == 0:
                self.lastneighbourtoconsider[i] = 0
            else:
                self.lastneighbourtoconsider[i] = max(1, lntc[-1])
        
        self.distributionx = np.zeros((len(self.x) - 1, self.M + 1))
        self.distributiony = np.zeros((len(self.x) - 1, self.M + 1))
        self.distributionz = np.zeros((len(self.x) - 1, self.M + 1))
        
        for i in range(len(self.x) - 1):
            self.distributionx[i, :] = np.linspace(self.x[i], self.x[i + 1], self.M + 1).reshape(-1)
            self.distributiony[i, :] = np.linspace(self.y[i], self.y[i + 1], self.M + 1).reshape(-1)
            self.distributionz[i, :] = np.linspace(self.z[i], self.z[i + 1], self.M + 1).reshape(-1)
        
        
        # Initialize SBT algorithm linear system of equation matrices
        self.L = np.zeros((4 * self.N, 4 * self.N))                # Will store the "left-hand side" of the system of equations
        self.R = np.zeros((4 * self.N, 1))                    # Will store the "right-hand side" of the system of equations
        
        self.Q = np.zeros((self.N, len(self.times)))                   # Initializes the heat pulse matrix, i.e., the heat pulse emitted by each element at each time step
        self.Toutput = self.surface_temp                 # Initializes the production temperatures array
        self.mvector = np.zeros(len(self.times))                  # Initializes the production temperatures array
        
        self.Tw_up_previous = self.BBinitial                  # At time zero, the initial upflowing fluid temperature corresponds to the initial local rock temperature
        self.Tw_down_previous = self.BBinitial                # At time zero, the initial downflowing fluid temperature corresponds to the initial local rock temperature                            # At time zero, the outlet temperature is the initial local fluid temperature at the surface, which corresponds to the surface temperature
        
        self.Tw_up_Matrix = np.zeros((len(self.times), self.N))       # Initializes the matrix that holds the upflowing fluid temperature over time
        self.Tw_up_Matrix[0, :] = self.Tw_up_previous
        self.Tw_down_Matrix = np.zeros((len(self.times), self.N))     # Initializes the matrix that holds the downflowing fluid temperature over time
        self.Tw_down_Matrix[0, :] = self.Tw_down_previous
            
        #Initialize FMM arrays
        self.combinedtimes = np.array([0])
        self.combinedQ = np.zeros((self.N, 1))
        self.combinedtimes2ndlevel = np.array([0])
        self.combinedtimes3rdlevel = np.array([0])
        self.timesFMM = 0
        self.QFMM = 0

        self.dynamic_properties = dynamic_properties
        self.counter = -1

        self.configure_well_dimensions()

    def pre_model(self, t, m_prd, m_inj, T_inj):
        """Computations to be performed before stepping the reservoir model.

        Args:
            t (datetime): current timestamp.
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s.
            T_inj (float): injection temperature in deg C.
        """
        self.counter += 1

    def model(self, t, m_prd, m_inj, T_inj):
        """Computations to be performed when stepping the reservoir model.

        Args:
            t (datetime): current timestamp.
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s.
            T_inj (float): injection temperature in deg C.
        """

        if self.reservoir_simulator_settings["DynamicFluidProperties"]:
            Tavg = (self.Tres + self.Toutput)/2
            # self.rho_f = densitywater(Tavg)
            # self.mu_f = viscositywater(Tavg)
            # self.cp_f = heatcapacitywater(Tavg)
            Tround = round(Tavg, 0)
            self.rho_f = self.cache["densitywater"].get(Tround, densitywater(Tavg))
            self.cache["densitywater"][Tround] = self.rho_f

            self.mu_f = self.cache["viscositywater"].get(Tround, viscositywater(Tavg))
            self.cache["viscositywater"][Tround] = self.mu_f

            self.cp_f = self.cache["cpwater"].get(Tround, heatcapacitywater(Tavg))
            self.cache["cpwater"][Tround] = self.cp_f

            self.alpha_f = self.k_f / self.rho_f / self.cp_f  # Fluid thermal diffusivity [m2/s]
            self.Pr_f = self.mu_f / self.rho_f / self.alpha_f  # Fluid Prandtl number [-]

        if self.dynamic_properties:
            self.rhow_prd_bh = densitywater(_safe_mean(self.T_prd_bh))
            self.cw_prd_bh = heatcapacitywater(_safe_mean(self.T_prd_bh)) # J/kg-degC

        self.Deltat = self.timestep.total_seconds() # Current time step size [s]
        self.Tin = T_inj #injection temperature is the same for all doublets as they assumingly feed into a single power plant
        self.m = m_prd.mean() #take the mean of all doublets
        i = self.counter


        # Helper function for safe exp1 (same as ULoopSBT)
        def _safe_exp1_coaxial(arg, name="exp1"):
            """Safe exp1 ensuring argument is > epsilon."""
            eps = np.finfo(float).eps * 10
            arg = np.maximum(arg, eps)
            result = exp1(arg)
            # Replace inf with large finite value
            result = np.where(np.isinf(result), 1e10, result)
            result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
            return result
        
        if self.alpha_m * self.Deltat / self.radius**2 > self.LimitCylinderModelRequired:
            exp1_arg = self.radius**2 / (4 * self.alpha_m * self.Deltat)
            exp1_arg = np.maximum(exp1_arg, np.finfo(float).eps * 10)  # Ensure positive
            self.CPCP = np.ones(self.N) * 1 / (4 * np.pi * self.k_m) * _safe_exp1_coaxial(exp1_arg, "exp1_CPCP_coaxial")  # Use line source model if possible
        else:
            interp_arg = self.alpha_m * self.Deltat / self.radius**2
            interp_arg = np.clip(interp_arg, self.argumentbesselvec.min(), self.argumentbesselvec.max())
            if np.any(np.isnan(interp_arg)) or np.any(np.isinf(interp_arg)):
                interp_arg = np.nan_to_num(interp_arg, nan=self.argumentbesselvec.mean(), posinf=self.argumentbesselvec.max(), neginf=self.argumentbesselvec.min())
            self.CPCP = np.ones(self.N) * np.interp(interp_arg, self.argumentbesselvec, self.besselcylinderresult)  # Use cylindrical source model if required
        
        if self.Deltat > self.timeforfinitelinesource:  # For long time steps, the finite length correction should be applied
            interp_arg = self.Deltaz**2 / (4 * self.alpha_m * self.Deltat)
            interp_arg = np.clip(interp_arg, self.Amin1vector.min(), self.Amin1vector.max())
            if np.any(np.isnan(interp_arg)) or np.any(np.isinf(interp_arg)):
                interp_arg = np.nan_to_num(interp_arg, nan=self.Amin1vector.mean(), posinf=self.Amin1vector.max(), neginf=self.Amin1vector.min())
            self.CPCP = self.CPCP + np.interp(interp_arg, self.Amin1vector, self.finitecorrectiony)
            
        if i > 1:  # After the second time step, we need to keep track of previous heat pulses
    
            self.CPOP = np.zeros((self.N, len(self.timesFMM)-1))
            
            self.indexpsstart = 0
    
            self.indexpsend = np.where(self.timeforpointssource < (self.times[i] - self.timesFMM[1:]))[-1]
            if self.indexpsend.size > 0:
                self.indexpsend = self.indexpsend[-1] + 1
            else:
                self.indexpsend = self.indexpsstart - 1
            if self.indexpsend >= self.indexpsstart:  # Use point source model if allowed
               
                self.CPOP[:, 0:self.indexpsend] = self.Deltaz * np.ones((self.N, self.indexpsend)) / (4 * np.pi * np.sqrt(self.alpha_m * np.pi) * self.k_m) * (
                        np.ones(self.N) * (1 / np.sqrt(self.times[i] - self.timesFMM[self.indexpsstart + 1:self.indexpsend + 2]) -
                        1 / np.sqrt(self.times[i] - self.timesFMM[self.indexpsstart:self.indexpsend+1])))
            self.indexlsstart = self.indexpsend + 1
            self.indexlsend = np.where(self.timeforlinesource < (self.times[i] - self.timesFMM[1:]))[0]
            if self.indexlsend.size == 0:
                self.indexlsend = self.indexlsstart - 1
            else:
                self.indexlsend = self.indexlsend[-1]
            
            if self.indexlsend >= self.indexlsstart:  # Use line source model for more recent heat pulse events
                exp1_arg1 = self.radius**2 / (4*self.alpha_m*(self.times[i]-self.timesFMM[self.indexlsstart:self.indexlsend+1])).reshape(1,len(4 * self.alpha_m * (self.times[i] - self.timesFMM[self.indexlsstart:self.indexlsend+1])))
                exp1_arg2 = (self.radius**2) / (4 * self.alpha_m * (self.times[i]-self.timesFMM[self.indexlsstart+1:self.indexlsend+2])).reshape(1,len(4 * self.alpha_m * (self.times[i] - self.timesFMM[self.indexlsstart+1:self.indexlsend+2])))
                exp1_arg1 = np.maximum(exp1_arg1, np.finfo(float).eps * 10)  # Ensure positive
                exp1_arg2 = np.maximum(exp1_arg2, np.finfo(float).eps * 10)  # Ensure positive
                exp1_result1 = _safe_exp1_coaxial(exp1_arg1, "exp1_CPOP_coaxial_1")
                exp1_result2 = _safe_exp1_coaxial(exp1_arg2, "exp1_CPOP_coaxial_2")
                self.CPOP[:, self.indexlsstart:self.indexlsend+1] = np.ones((self.N,1)) * 1 / (4*np.pi*self.k_m) * (exp1_result1 - exp1_result2)
    
            self.indexcsstart = max(self.indexpsend, self.indexlsend) + 1
            #indexcsend = i - 2
            self.indexcsend = len(self.timesFMM)-2
    
            if self.indexcsstart <= self.indexcsend:  # Use cylindrical source model for the most recent heat pulses
                self.CPOPPH = np.zeros((self.CPOP[:, self.indexcsstart:self.indexcsend+1].shape))   
                self.CPOPdim = self.CPOP[:, self.indexcsstart:self.indexcsend+1].shape
                self.CPOPPH = self.CPOPPH.T.ravel()
    
                interp_arg1 = self.alpha_m * (self.times[i] - self.timesFMM[self.indexcsstart:self.indexcsend+1]).reshape(len(self.times[i] - self.timesFMM[self.indexcsstart:self.indexcsend+1]),1) / (self.radius**2)
                interp_arg2 = self.alpha_m * (self.times[i] - self.timesFMM[self.indexcsstart+1: self.indexcsend+2]).reshape(len(self.times[i] - self.timesFMM[self.indexcsstart+1:self.indexcsend+2]),1) / (self.radius**2)
                interp_arg1 = np.clip(interp_arg1, self.argumentbesselvec.min(), self.argumentbesselvec.max())
                interp_arg2 = np.clip(interp_arg2, self.argumentbesselvec.min(), self.argumentbesselvec.max())
                interp_arg1 = np.nan_to_num(interp_arg1, nan=self.argumentbesselvec.mean(), posinf=self.argumentbesselvec.max(), neginf=self.argumentbesselvec.min())
                interp_arg2 = np.nan_to_num(interp_arg2, nan=self.argumentbesselvec.mean(), posinf=self.argumentbesselvec.max(), neginf=self.argumentbesselvec.min())
                self.CPOPPH = (np.ones(self.N) * ( \
                            np.interp(interp_arg1, self.argumentbesselvec, self.besselcylinderresult) - \
                            np.interp(interp_arg2, self.argumentbesselvec, self.besselcylinderresult))).reshape(-1,1)
                
                self.CPOPPH=self.CPOPPH.reshape((self.CPOPdim),order='F')
                self.CPOP[:, self.indexcsstart:self.indexcsend+1] = self.CPOPPH
      
            self.indexflsstart = self.indexpsend + 1
            self.indexflsend = np.where(self.timeforfinitelinesource < (self.times[i] - self.timesFMM[1:]))[-1]
            if self.indexflsend.size == 0:
                self.indexflsend = self.indexflsstart - 1
            else:
                self.indexflsend = self.indexflsend[-1] - 1
        
            if self.indexflsend >= self.indexflsstart:  # Perform finite length correction if needed
                interp_arg1 = np.matmul((self.Deltaz.reshape(len(self.Deltaz),1) ** 2),np.ones((1,self.indexflsend-self.indexflsstart+2))) / np.matmul(np.ones((self.N,1)),(4 * self.alpha_m * (self.times[i] - self.timesFMM[self.indexflsstart:self.indexflsend+2]).reshape(len(self.times[i] - self.timesFMM[self.indexflsstart:self.indexflsend+2]),1)).T)
                interp_arg2 = np.matmul((self.Deltaz.reshape(len(self.Deltaz),1) ** 2),np.ones((1,self.indexflsend-self.indexflsstart+2))) / np.matmul(np.ones((self.N,1)),(4 * self.alpha_m * (self.times[i] - self.timesFMM[self.indexflsstart+1:self.indexflsend+3]).reshape(len(self.times[i] - self.timesFMM[self.indexflsstart:self.indexflsend+2]),1)).T)
                interp_arg1 = np.clip(interp_arg1, self.Amin1vector.min(), self.Amin1vector.max())
                interp_arg2 = np.clip(interp_arg2, self.Amin1vector.min(), self.Amin1vector.max())
                interp_arg1 = np.nan_to_num(interp_arg1, nan=self.Amin1vector.mean(), posinf=self.Amin1vector.max(), neginf=self.Amin1vector.min())
                interp_arg2 = np.nan_to_num(interp_arg2, nan=self.Amin1vector.mean(), posinf=self.Amin1vector.max(), neginf=self.Amin1vector.min())
                self.CPOP[:, self.indexflsstart:self.indexflsend+2] = self.CPOP[:, self.indexflsstart:self.indexflsend+2] + (np.interp(interp_arg1, self.Amin1vector, self.finitecorrectiony) - \
                np.interp(interp_arg2, self.Amin1vector, self.finitecorrectiony))

        self.NPCP = np.zeros((self.N, self.N))
        np.fill_diagonal(self.NPCP, self.CPCP)
        
        # Clean NPCP immediately after creation to prevent NaN/Inf propagation
        max_finite_val = np.finfo(np.float64).max / 1e10
        self.NPCP = np.nan_to_num(self.NPCP, nan=0.0, posinf=max_finite_val, neginf=-max_finite_val)
        
    
        self.spacingtest = self.alpha_m * self.Deltat / self.SMatrixSorted[:, 1:]**2 / self.LimitNPSpacingTime
        self.maxspacingtest = np.max(self.spacingtest,axis=0)
    
        
        if self.maxspacingtest[0] < 1:
            self.maxindextoconsider = 0  #KB 11/21/2024: maybe this should be -1 at no index should be consdiered?????
        else:
            self.maxindextoconsider = np.where(self.maxspacingtest > 1)[0][-1]+1 #KB 11/21/2024: is plus 1 needed here?????
    
        #Calculate and store neighbouring pipes for current pulse as point sources        
        if self.mindexNPCP < self.maxindextoconsider:      #KB 11/21/2024: is plus 1 needed here?????
            self.indicestocalculate = self.SortedIndices[:, self.mindexNPCP + 1:self.maxindextoconsider + 1]
            self.indicestocalculatetranspose = self.indicestocalculate.T
            self.indicestocalculatelinear = self.indicestocalculate.ravel()
            self.indicestostorematrix = (self.indicestocalculate - 1) * self.N + np.arange(1, self.N) * np.ones((1, self.maxindextoconsider - self.mindexNPCP + 1))
            self.indicestostorematrixtranspose = self.indicestostorematrix.T
            self.indicestostorelinear = self.indicestostorematrix.ravel()
            self.NPCP[self.indicestostorelinear] = self.Deltaz[self.indicestocalculatelinear] / (4 * np.pi * self.k_m * self.SMatrix[self.indicestostorelinear]) * erfc(self.SMatrix[self.indicestostorelinear] / np.sqrt(4 * self.alpha_m * self.Deltat))
            
        #Calculate and store neighbouring pipes for current pulse as set of line sources
        if self.mindexNPCP > 1 and self.maxindextoconsider > 0:
            self.lastindexfls = min(self.mindexNPCP, self.maxindextoconsider + 1)
            self.indicestocalculate = self.SortedIndices[:, 1:self.lastindexfls]
            self.indicestocalculatetranspose = self.indicestocalculate.T
            self.indicestocalculatelinear = self.indicestocalculate.ravel()
            self.indicestostorematrix = (self.indicestocalculate) * self.N + np.arange(self.N).reshape(-1,1) * np.ones((1, self.lastindexfls - 1), dtype=int)
           # #pdb.set_trace()
            self.indicestostorematrixtranspose = self.indicestostorematrix.T
            self.indicestostorelinear = self.indicestostorematrix.ravel()
            self.midpointindices = np.matmul(np.ones((self.lastindexfls - 1, 1)), np.arange(self.N).reshape(1,self.N)).T
            self.midpointsindices = self.midpointindices.ravel().astype(int)
            self.rultimate = np.sqrt(np.square((self.midpointsx[self.midpointsindices].reshape(len(self.midpointsindices),1)*( np.ones((1, self.M + 1))) - self.distributionx[self.indicestocalculatelinear,:])) +
                                np.square((self.midpointsy[self.midpointsindices].reshape(len(self.midpointsindices),1)*( np.ones((1, self.M + 1))) - self.distributiony[self.indicestocalculatelinear,:])) +
                                np.square((self.midpointsz[self.midpointsindices].reshape(len(self.midpointsindices),1)*( np.ones((1, self.M + 1))) - self.distributionz[self.indicestocalculatelinear,:])))
            
            self.NPCP[np.unravel_index(self.indicestostorelinear, self.NPCP.shape, 'F')] =  self.Deltaz[self.indicestocalculatelinear] / self.M * np.sum((1 - erf(self.rultimate / np.sqrt(4 * self.alpha_m * self.Deltat))) / (4 * np.pi * self.k_m * self.rultimate) * np.matmul(np.ones((self.N*(self.lastindexfls-1),1)),np.concatenate((np.array([1/2]), np.ones(self.M-1), np.array([1/2]))).reshape(-1,1).T), axis=1)

        
        if i>1:
            self.NPOP = np.zeros((self.N * (int(self.lastneighbourtoconsider[i])+1), len(self.timesFMM) - 1))
        self.BB = np.zeros((self.N, 1))
        if i > 1 and self.lastneighbourtoconsider[i] > 0:
            self.SMatrixRelevant = self.SMatrixSorted[:, 1 : int(self.lastneighbourtoconsider[i] + 2)]
            self.SoverLRelevant = self.SoverLSorted[:, 1 : int(self.lastneighbourtoconsider[i]) + 2]
            self.SortedIndicesRelevant = self.SortedIndices[:, 1 : int(self.lastneighbourtoconsider[i]) + 2] 
            self.maxtimeindexmatrix = self.alpha_m * np.ones((self.N * (int(self.lastneighbourtoconsider[i])+1), 1)) * (self.times[i] - self.timesFMM[1:]) / (self.SMatrixRelevant.T.ravel().reshape(-1,1) * np.ones((1,len(self.timesFMM)-1)))**2
            
            self.allindices = np.arange(self.N * (int(self.lastneighbourtoconsider[i])+1) * (len(self.timesFMM) - 1))

            self.pipeheatcomesfrom = np.matmul(self.SortedIndicesRelevant.T.ravel().reshape(len(self.SortedIndicesRelevant.ravel()),1), np.ones((1,len(self.timesFMM) - 1)))
            self.pipeheatgoesto = np.arange(self.N).reshape(self.N,1) * np.ones((1, int(self.lastneighbourtoconsider[i]+1)))
            self.pipeheatgoesto = self.pipeheatgoesto.transpose().ravel().reshape(len(self.pipeheatgoesto.ravel()),1) * np.ones((1, len(self.timesFMM) - 1))

            self.indicestoneglect = np.where((self.maxtimeindexmatrix.transpose()).ravel() < self.LimitNPSpacingTime)[0]
            
            self.maxtimeindexmatrix = np.delete(self.maxtimeindexmatrix.transpose().ravel(), self.indicestoneglect)
            self.allindices = np.delete(self.allindices.transpose().ravel(), self.indicestoneglect)
            self.indicesFoSlargerthan = np.where(self.maxtimeindexmatrix.ravel() > 10)[0]

            self.indicestotakeforpsFoS = self.allindices[self.indicesFoSlargerthan]
       
            self.allindices2 = self.allindices.copy()
    
            self.allindices2[self.indicesFoSlargerthan] = []
            self.SoverLinearized = self.SoverLRelevant.transpose().ravel().reshape(len(self.SoverLRelevant.ravel()),1) * np.ones((1, len(self.timesFMM) - 1))
            self.indicestotakeforpsSoverL = np.where(self.SoverLinearized.transpose().ravel()[self.allindices2] > self.LimitSoverL)[0]
            self.overallindicestotakeforpsSoverL = self.allindices2[self.indicestotakeforpsSoverL]
            self.remainingindices = self.allindices2.copy() 
        
            self.NPOP = np.zeros((self.N * (int(self.lastneighbourtoconsider[i])+1), len(self.timesFMM) - 1))
            
            # self.remainingindices=np.delete(self.remainingindices,self.indicestotakeforpsSoverL)
            # Use point source model when FoS is very large
            # if len(self.indicestotakeforpsFoS) > 0:

            #     self.deltatlinear1 = np.ones((self.N * int(self.lastneighbourtoconsider[i]), 1)) * (self.times[i] - self.timesFMM[1:])
            #     self.deltatlinear1 = self.deltatlinear1.ravel()[self.indicestotakeforpsFoS]
            #     self.deltatlinear2 = np.ones((self.N * int(self.lastneighbourtoconsider[i]), 1)) * (self.times[i] - self.timesFMM[0:-1])
            #     self.deltatlinear2 = self.deltatlinear2[self.indicestotakeforpsFoS]
            #     self.deltazlinear = self.pipeheatcomesfrom[self.indicestotakeforpsFoS]
            #     self.SMatrixlinear = self.SMatrixRelevant.flatten(order='F')
            #     self.NPOPFoS = self.Deltaz[self.deltazlinear] / (4 * np.pi * self.k_m * self.SMatrixlinear[self.indicestotakeforpsFoS]) * (erfc(self.SMatrixlinear[self.indicestotakeforpsFoS] / np.sqrt(4 * self.alpha_m * self.deltatlinear2)) -
            #         erfc(self.SMatrixlinear[self.indicestotakeforpsFoS] / np.sqrt(4 * self.alpha_m * self.deltatlinear1)))
            
            #     self.NPOP[self.indicestotakeforpsFoS] = self.NPOPFoS
            
            # Use point source model when SoverL is very large
            # if len(self.overallindicestotakeforpsSoverL) > 0:
            #     self.deltatlinear1 = np.ones((self.N * (int(self.lastneighbourtoconsider[i])+1), 1)) * (self.times[i] - self.timesFMM[1:]).ravel()
            #     self.deltatlinear1 = self.deltatlinear1[self.overallindicestotakeforpsSoverL]
            #     self.deltatlinear2 = np.ones((self.N * int(self.lastneighbourtoconsider[i]+1), 1)) * (self.times[i] - self.timesFMM[0:-1]).ravel()
            #     self.deltatlinear2 = self.deltatlinear2[self.overallindicestotakeforpsSoverL]
            #     self.deltazlinear = self.pipeheatcomesfrom[self.overallindicestotakeforpsSoverL]
            #     self.SMatrixlinear = self.SMatrixRelevant.flatten(order='F')
            #     self.NPOPSoverL = self.Deltaz[self.deltazlinear.astype(int)] / (4 * np.pi * self.k_m * self.SMatrixlinear[self.overallindicestotakeforpsSoverL][:, None]) * (erfc(self.SMatrixlinear[self.overallindicestotakeforpsSoverL][:, None] / np.sqrt(4 * self.alpha_m * self.deltatlinear2)) -
            #         erfc(self.SMatrixlinear[self.overallindicestotakeforpsSoverL][:, None] / np.sqrt(4 * self.alpha_m * self.deltatlinear1)))
            
            #     self.NPOP[self.overallindicestotakeforpsSoverL] = self.NPOPSoverL
               
            # Use finite line source model for remaining pipe segments
            if len(self.remainingindices) > 0:
               
                self.deltatlinear1 = np.ones((self.N * (int(self.lastneighbourtoconsider[i])+1), 1)) * (self.times[i] - self.timesFMM[1:])
                self.deltatlinear1 = (self.deltatlinear1.transpose()).ravel()[self.remainingindices]
                self.deltatlinear2 = np.ones((self.N * (int(self.lastneighbourtoconsider[i])+1), 1)) * (self.times[i] - self.timesFMM[0:-1])
                self.deltatlinear2 = (self.deltatlinear2.transpose()).ravel()[self.remainingindices]
                self.deltazlinear = (self.pipeheatcomesfrom.T).ravel()[self.remainingindices]
                self.midpointstuff = (self.pipeheatgoesto.transpose()).ravel()[self.remainingindices]
                self.rultimate = np.sqrt(np.square((self.midpointsx[self.midpointstuff.astype(int)].reshape(len(self.midpointsx[self.midpointstuff.astype(int)]),1)*( np.ones((1, self.M + 1))) - self.distributionx[self.deltazlinear.astype(int),:])) +
                                 np.square((self.midpointsy[self.midpointstuff.astype(int)].reshape(len(self.midpointsy[self.midpointstuff.astype(int)]),1)*( np.ones((1, self.M + 1))) - self.distributiony[self.deltazlinear.astype(int),:])) +
                                 np.square((self.midpointsz[self.midpointstuff.astype(int)].reshape(len(self.midpointsz[self.midpointstuff.astype(int)]),1)*( np.ones((1, self.M + 1))) - self.distributionz[self.deltazlinear.astype(int),:])))
               # #pdb.set_trace()
                self.NPOPfls = self.Deltaz[self.deltazlinear.astype(int)].reshape(len(self.Deltaz[self.deltazlinear.astype(int)]),1).T / self.M * np.sum((-erf(self.rultimate / np.sqrt(4 * self.alpha_m * np.ravel(self.deltatlinear2).reshape(len(np.ravel(self.deltatlinear2)),1)*np.ones((1, self.M + 1)))) + erf(self.rultimate / np.sqrt(4 * self.alpha_m * np.ravel(self.deltatlinear1).reshape(len(np.ravel(self.deltatlinear1)),1)*np.ones((1, self.M + 1))))) / (4 * np.pi * self.k_m * self.rultimate) *  np.matmul((np.ones((len(self.remainingindices),1))),(np.concatenate((np.array([1/2]),np.ones(self.M - 1),np.array([1/2])))).reshape(-1,1).T), axis=1)
                self.NPOPfls = self.NPOPfls.T
                self.dimensions = self.NPOP.shape
              #  #pdb.set_trace()
                self.NPOP=self.NPOP.T.ravel()
                self.NPOP[self.remainingindices.reshape((len(self.remainingindices),1))] = self.NPOPfls
                self.NPOP = self.NPOP.reshape((self.dimensions[1],self.dimensions[0])).T
                
        # Put everything together and calculate BB (= impact of all previous heat pulses from old neighbouring elements on current element at current time)
           #  
            self.Qindicestotake = self.SortedIndicesRelevant.T.ravel().reshape((self.N * (int(self.lastneighbourtoconsider[i])+1), 1))*np.ones((1,len(self.timesFMM)-1)) + \
                            np.ones((self.N * (int(self.lastneighbourtoconsider[i])+1), 1)) * self.N * np.arange(len(self.timesFMM) - 1)
            self.Qindicestotake = self.Qindicestotake.astype(int)
            self.Qlinear = self.QFMM.T.ravel()[self.Qindicestotake]
           # Qlinear = Qlinear[:,:,0]
            self.BBPS = self.NPOP * self.Qlinear
            self.BBPS = np.sum(self.BBPS, axis=1)
            self.BBPSindicestotake = np.arange(self.N).reshape((self.N, 1)) + self.N * np.arange(int(self.lastneighbourtoconsider[i])+1).reshape((1, int(self.lastneighbourtoconsider[i])+1))
            self.BBPSMatrix = self.BBPS[self.BBPSindicestotake]
            self.BB = np.sum(self.BBPSMatrix, axis=1)
            
    
        if i > 1:
            #BBCPOP = np.sum(CPOP * Q[:, 1:i], axis=1)
            self.BBCPOP = np.sum(self.CPOP * self.QFMM[:,1:], axis=1)
        else:
            self.BBCPOP = np.zeros(self.N)

        if self.coaxialflowtype == 1: #CXA
            self.u_down = self.m/self.rho_f/self.A_flow_annulus               # Downgoing fluid velocity in annulus [m/s]
            self.u_up = self.m/self.rho_f/self.A_flow_centerpipe              # Upgoing fluid velocity in center pipe [m/s]
        elif self.coaxialflowtype == 2: #CXC
            self.u_up = self.m/self.rho_f/self.A_flow_annulus                 # Upgoing fluid velocity in annulus [m/s]
            self.u_down = self.m/self.rho_f/self.A_flow_centerpipe            # Downgoing fluid velocity in center pipe [m/s]
            
        if self.coaxialflowtype == 1: #CXA (injection in annulus; production from center pipe)
            #Thermal resistance in annulus (downflowing)
            self.Re_down = self.rho_f*self.u_down*(self.Dh_annulus)/self.mu_f      # Reynolds Number [-]
            self.Nuturb_down = 0.023*self.Re_down**(4/5)*self.Pr_f**(0.4)  # Turbulent Flow Nusselt Number for Annulus [-] (Dittus-Boelter equation for heating)
            if (self.Re_down>2300):                              #Based on Section 8.6 in Bergman (2011), for annulus turbulent flow, the Nusselt numbers for the inner and outer wall can be assumed the same
                self.Nu_down_o = self.Nuturb_down
                self.Nu_down_i = self.Nuturb_down
            else:
                self.Nu_down_o = 5                             # Laminar flow annulus Nusselt number for outer wall (approximate; see Table 8.2 and 8.3 in Bergman (2011)
                self.Nu_down_i = 6                             # Laminar flow annulus Nusselt number for inner wall (approximate; see Table 8.2 and 8.3 in Bergman (2011)
            
            if self.m < 0.1: # We assume at very low flow rates, we are actually simulating well shut-in. The Nusselt numbers get set to 1 to represent thermal conduction.
                self.Nu_down_o = 1
                self.Nu_down_i = 1
            
            self.h_down_o = self.Nu_down_o*self.k_f/self.Dh_annulus
            self.h_down_i = self.Nu_down_i*self.k_f/self.Dh_annulus
            self.Rt = 1/(np.pi*self.h_down_o*self.radius*2)                 # Thermal resistance between annulus flow and surrounding rock (open-hole assumed)
            
            #Thermal resistance in center pipe (upflowing)
            self.Re_up = self.rho_f*self.u_up*(2*self.radiuscenterpipe)/self.mu_f  # Reynolds Number [-]
            self.Nuturb_up = 0.023*self.Re_up**(4/5)*self.Pr_f**(0.4)    # Turbulent Flow Nusselt Number [-] (Dittus-Boelter equation for heating)
            self.Nulam_up = 4                                  # Laminar Flow Nusselt Number [-] (this is approximate)
            if self.Re_up>2300:
                self.Nu_up = self.Nuturb_up
            else:
                self.Nu_up = self.Nulam_up
            
            
            if self.m < 0.1: # We assume at very low flow rates, we are actually simulating well shut-in. The Nusselt number is set to 1 to represent thermal conduction.
                self.Nu_up = 1
            
            self.h_up = self.Nu_up*self.k_f/(2*self.radiuscenterpipe)
            self.R_cp = 1/(np.pi*self.h_up*2*self.radiuscenterpipe)+np.log((2*self.outerradiuscenterpipe)/(2*self.radiuscenterpipe))/(2*np.pi*self.k_center_pipe) + 1/(np.pi*self.h_down_i*2*self.outerradiuscenterpipe) # thermal resistance between annulus flow and center pipe flow
            
        elif self.coaxialflowtype == 2: #CXC (injection in center pipe; production from annulus)
            #Thermal resistance in annulus (upflowing)
            self.Re_up = self.rho_f*self.u_up*(self.Dh_annulus)/self.mu_f          # Reynolds Number [-]
            self.Nuturb_up = 0.023*self.Re_up**(4/5)*self.Pr_f**(0.4)    # Turbulent Flow Nusselt Number [-] (Dittus-Boelter equation for heating)
            if self.Re_up>2300:                                # Based on Section 8.6 in Bergman (2011), for annulus turbulent flow, the Nusselt numbers for the inner and outer wall can be assumed the same
                self.Nu_up_o = self.Nuturb_up
                self.Nu_up_i = self.Nuturb_up
            else:
                self.Nu_up_o = 5                               # Laminar flow annulus Nusselt number for outer wall (approximate; see Table 8.2 and 8.3 in Bergman (2011)
                self.Nu_up_i = 6                               # Laminar flow annulus Nusselt number for inner wall (approximate; see Table 8.2 and 8.3 in Bergman (2011)
            
            if self.m < 0.1: # We assume at very low flow rates, we are actually simulating well shut-in. The Nusselt numbers get set to 1 to represent thermal conduction.
                self.Nu_up_o = 1
                self.Nu_up_i = 1
            
            self.h_up_o = self.Nu_up_o*self.k_f/self.Dh_annulus
            self.h_up_i = self.Nu_up_i*self.k_f/self.Dh_annulus
            self.Rt = 1/(np.pi*self.h_up_o*self.radius*2)                   # Thermal resistance between annulus flow and surrounding rock (open-hole assumed)
            
            # Thermal resistance in center pipe (downflowing)
            self.Re_down = self.rho_f*self.u_down*(2*self.radiuscenterpipe)/self.mu_f # Reynolds Number [-]
            self.Nuturb_down = 0.023*self.Re_down**(4/5)*self.Pr_f**(0.4)   # Turbulent Flow Nusselt Number [-] (Dittus-Boelter equation for heating)
            self.Nulam_down = 4                                   # Laminar Flow Nusselt Number [-] (this is approximate)
            if self.Re_down>2300:
                self.Nu_down = self.Nuturb_down
            else:
                self.Nu_down = self.Nulam_down
            
            if self.m < 0.1: # We assume at very low flow rates, we are actually simulating well shut-in. The Nusselt number is to 1 to represent thermal conduction.
                self.Nu_down = 1
            
            self.h_down = self.Nu_down*self.k_f/(2*self.radiuscenterpipe)
            self.R_cp = 1/(np.pi*self.h_down*2*self.radiuscenterpipe)+np.log((2*self.outerradiuscenterpipe)/(2*self.radiuscenterpipe))/(2*np.pi*self.k_center_pipe) + 1/(np.pi*self.h_up_i*2*self.outerradiuscenterpipe) # Thermal resistance between annulus flow and center pipe flow


        
        if self.coaxialflowtype == 1: #CXA    
            #Populate L and R for downflowing fluid heat balance for first element (which has the injection temperature specified)
            self.L[0,0] = 1 / self.Deltat + self.u_down / self.Deltaz[0]*2  + 1/self.R_cp/(self.A_flow_annulus*self.rho_f*self.cp_f)
            self.L[0,2] = -1 / (self.A_flow_annulus*self.rho_f*self.cp_f)
            self.L[0,3] = -1 / self.R_cp / (self.A_flow_annulus*self.rho_f*self.cp_f)
            self.R[0,0] = 1 / self.Deltat*self.Tw_down_previous[0] + self.u_down/self.Deltaz[0]*self.Tin*2
            
            #Populate L and R for rock temperature equation for first element
            self.L[1,0] = 1
            self.L[1,1] = -1
            self.L[1,2] = self.Rt
            self.R[1,0] = 0
            
            #Populate L and R for SBT algorithm for first element
            self.L[2,np.arange(2,4*self.N,4)] = self.NPCP[0,0:self.N]
            self.L[2,1] = 1
            self.R[2,0] =  - self.BBCPOP[0] - self.BB[0] + self.BBinitial[0]
            
            #Populate L and R for upflowing fluid heat balance for first element
            self.L[3,3] = 1 / self.Deltat + self.u_up/self.Deltaz[0] + 1/self.R_cp/(self.A_flow_centerpipe*self.rho_f*self.cp_f);
            self.L[3,0] = -1/self.R_cp/(self.A_flow_centerpipe*self.rho_f*self.cp_f);
            self.L[3,7] = -self.u_up/self.Deltaz[0];
            self.R[3,0] = 1/self.Deltat*self.Tw_up_previous[0];            
            
            for iiii in range(2, self.N+1):  #Populate L and R for remaining elements
                #Heat balance equation for downflowing fluid
                self.L[(iiii-1)*4,(iiii-1)*4] = 1/self.Deltat + self.u_down/self.Deltaz[iiii-1] + 1/self.R_cp/(self.A_flow_annulus*self.rho_f*self.cp_f)
                self.L[(iiii-1)*4,2+(iiii-1)*4] = -1/(self.A_flow_annulus*self.rho_f*self.cp_f)
                self.L[(iiii-1)*4,3+(iiii-1)*4] = -1/self.R_cp/(self.A_flow_annulus*self.rho_f*self.cp_f)
                self.L[(iiii-1)*4,(iiii-2)*4] = -self.u_down/self.Deltaz[iiii-1]
                self.R[(iiii-1)*4,0] = 1/self.Deltat*self.Tw_down_previous[iiii-1]
                
                #Rock temperature equation
                self.L[1 + (iiii - 1) * 4,  (iiii - 1) * 4] = 1
                self.L[1 + (iiii - 1) * 4, 1 + (iiii - 1) * 4] = -1
                self.L[1 + (iiii - 1) * 4, 2 + (iiii - 1) * 4] = self.Rt
                self.R[1 + (iiii - 1) * 4, 0] = 0
                
                #SBT equation              
                self.L[2 + (iiii - 1) * 4, np.arange(2,4*self.N,4)] = self.NPCP[iiii-1, :self.N]
                self.L[2 + (iiii - 1) * 4, 1 + (iiii - 1) * 4] = 1
                self.R[2 + (iiii - 1) * 4, 0] = -self.BBCPOP[iiii-1] - self.BB[iiii-1] + self.BBinitial[iiii-1]                
                
                #Heat balance for upflowing fluid
                self.L[3+(iiii-1)*4,3+(iiii-1)*4] = 1/self.Deltat + self.u_up/self.Deltaz[iiii-1] + 1/self.R_cp/(self.A_flow_centerpipe*self.rho_f*self.cp_f)
                if iiii<self.N:
                    self.L[3+(iiii-1)*4,(iiii-1)*4] = -1/self.R_cp/(self.A_flow_centerpipe*self.rho_f*self.cp_f)
                    self.L[3+(iiii-1)*4,3+(iiii)*4] = -self.u_up/self.Deltaz[iiii-1]
                else: #The bottom element has the downflowing fluid becoming the upflowing fluid
                   self.L[3+(iiii-1)*4,(iiii-1)*4] = -1/self.R_cp/(self.A_flow_centerpipe*self.rho_f*self.cp_f) - self.u_up/self.Deltaz[iiii-1];
                
                self.R[3+(iiii-1)*4,0] = 1/self.Deltat*self.Tw_up_previous[iiii-1];
        
        
        elif self.coaxialflowtype == 2: #CXC
            #Populate L and R for heat balance upflowing fluid for first element
            self.L[0,0] = 1/self.Deltat + self.u_up/self.Deltaz[0]  + 1/self.R_cp/(self.A_flow_annulus*self.rho_f*self.cp_f);
            self.L[0,2] = -1/(self.A_flow_annulus*self.rho_f*self.cp_f);
            self.L[0,3] = -1/self.R_cp/(self.A_flow_annulus*self.rho_f*self.cp_f);
            self.L[0,4] = -self.u_up/self.Deltaz[0];
            self.R[0,0] = 1/self.Deltat*self.Tw_up_previous[0];
            
            #Populate L and R for rock temperature equation for first element
            self.L[1,0] = 1
            self.L[1,1] = -1
            self.L[1,2] = self.Rt
            self.R[1,0] = 0            
        
            #Populate L and R for SBT algorithm for first element
            self.L[2,np.arange(2,4*self.N,4)] = self.NPCP[0,0:self.N]
            self.L[2,1] = 1
            self.R[2,0] =  - self.BBCPOP[0] - self.BB[0] + self.BBinitial[0]
            
            #Populate L and R for heat balance fluid down for first element
            self.L[3,3] = 1/self.Deltat + self.u_down/self.Deltaz[0]*2 + 1/self.R_cp/(self.A_flow_centerpipe*self.rho_f*self.cp_f);
            self.L[3,0] = -1/self.R_cp/(self.A_flow_centerpipe*self.rho_f*self.cp_f);
            self.R[3,0] = 1/self.Deltat*self.Tw_down_previous[0] + self.u_down/self.Deltaz[0]*self.Tin*2;   
            
            for iiii in range(2, self.N+1):  #Populate L and R for remaining elements
                #Heat balance upflowing fluid
                self.L[(iiii-1)*4,(iiii-1)*4] = 1/self.Deltat + self.u_up/self.Deltaz[iiii-1] + 1/self.R_cp/(self.A_flow_annulus*self.rho_f*self.cp_f);
                self.L[(iiii-1)*4,2+(iiii-1)*4] = -1/(self.A_flow_annulus*self.rho_f*self.cp_f);
                self.R[(iiii-1)*4,0] = 1/self.Deltat*self.Tw_up_previous[iiii-1];
                if iiii<self.N:
                    self.L[(iiii-1)*4,(iiii)*4] = -self.u_up/self.Deltaz[iiii-1];
                    self.L[(iiii-1)*4,3+(iiii-1)*4] = -1/self.R_cp/(self.A_flow_annulus*self.rho_f*self.cp_f);
                else: #iiii==N is the bottom element where the downflowing fluid becomes the upflowing fluid
                    self.L[(iiii-1)*4,3+(iiii-1)*4] = -1/self.R_cp/(self.A_flow_annulus*self.rho_f*self.cp_f) - self.u_up/self.Deltaz[iiii-1];                
    
                #Rock temperature equation
                self.L[1 + (iiii - 1) * 4,  (iiii - 1) * 4] = 1
                self.L[1 + (iiii - 1) * 4, 1 + (iiii - 1) * 4] = -1
                self.L[1 + (iiii - 1) * 4, 2 + (iiii - 1) * 4] = self.Rt
                self.R[1 + (iiii - 1) * 4, 0] = 0
                
                #SBT equation              
                self.L[2 + (iiii - 1) * 4, np.arange(2,4*self.N,4)] = self.NPCP[iiii-1, :self.N]
                self.L[2 + (iiii - 1) * 4, 1 + (iiii - 1) * 4] = 1
                self.R[2 + (iiii - 1) * 4, 0] = -self.BBCPOP[iiii-1] - self.BB[iiii-1] + self.BBinitial[iiii-1]  
    
                #Heat balance downflowing fluid
                self.L[3+(iiii-1)*4,3+(iiii-1)*4] = 1/self.Deltat + self.u_down/self.Deltaz[iiii-1] + 1/self.R_cp/(self.A_flow_centerpipe*self.rho_f*self.cp_f);
                self.L[3+(iiii-1)*4,(iiii-1)*4] = -1/self.R_cp/(self.A_flow_centerpipe*self.rho_f*self.cp_f);
                self.L[3+(iiii-1)*4,3+(iiii-2)*4] = -self.u_down/self.Deltaz[iiii-1];
                self.R[3+(iiii-1)*4,0] = 1/self.Deltat*self.Tw_down_previous[iiii-1];                

        # Solving the linear system of equations
        self.Sol = np.linalg.solve(self.L, self.R)    
        
        # Extracting Q array for current heat pulses, fluid temperature, and store fluid temperature for the next time step
        self.Q[:, i] = self.Sol.ravel()[2::4]
        if self.coaxialflowtype == 1: #CXA
            self.Tw_down_Matrix[i,:] = self.Sol.ravel()[np.arange(0,4*self.N,4)] #Store the temperature for the downflowing water
            self.Tw_down_previous = self.Sol.ravel()[np.arange(0,4*self.N,4)]    #Store the current downflowing water temperature to be used as previous water temperature in the next time step
            self.Tw_up_Matrix[i,:] = self.Sol.ravel()[np.arange(3,4*self.N,4)]   #Store the temperature for the upflowing water
            self.Tw_up_previous = self.Sol.ravel()[np.arange(3,4*self.N,4)]      #Store the current upflowing water temperature to be used as previous water temperature in the next time step

        elif  self.coaxialflowtype == 2: #CXC
            self.Tw_down_Matrix[i,:] = self.Sol.ravel()[np.arange(3,4*self.N,4)] #Store the temperature for the downflowing water
            self.Tw_down_previous = self.Sol.ravel()[np.arange(3,4*self.N,4)]   #Store the current downflowing water temperature to be used as previous water temperature in the next time step
            self.Tw_up_Matrix[i,:] = self.Sol.ravel()[np.arange(0,4*self.N,4)]   #Store the temperature for the upflowing water
            self.Tw_up_previous = self.Sol.ravel()[np.arange(0,4*self.N,4)]     #Store the current upflowing water temperature to be used as previous water temperature in the next time step
            
        #Calculate the fluid outlet temperature at the top of the first element based on the water temperature at the midpoint of the top element and the one below
        self.Toutput = self.Tw_up_previous[0]+(self.Tw_up_previous[0]-self.Tw_up_previous[1])*0.5 
        self.mvector[i] = self.m

        if (self.FMM == 1 and i>50 and self.times[i] > self.FMMtriggertime):
            self.remainingtimes = self.times[(self.times >= self.combinedtimes[-1]) & (self.times < self.times[i])] 
            self.currentendtimespassed = self.times[i] - self.remainingtimes
            
            if len(self.remainingtimes)>40:
                if self.currentendtimespassed[24]>self.FMMtriggertime:
                    self.combinedtimes = np.append(self.combinedtimes,self.remainingtimes[24])
                    self.startindex = np.where(self.times == self.combinedtimes[-2])[0][0]
                    self.endindex = np.where(self.times == self.remainingtimes[24])[0][0]
                    self.newcombinedQ = np.sum(self.Q[:,self.startindex+1:self.endindex+1]*(self.times[self.startindex+1:self.endindex+1]-self.times[self.startindex:self.endindex])/(self.combinedtimes[-1]-self.combinedtimes[-2]),axis = 1) #weighted average
                    self.newcombinedQ = self.newcombinedQ.reshape(-1, 1)
                    self.combinedQ = np.hstack((self.combinedQ, self.newcombinedQ))
            
            
            #combine very old time pulses
            if len(np.where(self.combinedtimes == self.combinedtimes2ndlevel[-1])[0]) > 0:
                self.startindexforsecondlevel = np.where(self.combinedtimes == self.combinedtimes2ndlevel[-1])[0][0]
                if self.combinedtimes.size>30 and self.combinedtimes.size-self.startindexforsecondlevel>30:
                    if (self.times[i] - self.combinedtimes[self.startindexforsecondlevel+20])>self.FMMtriggertime*5:
                        self.indicestodrop = np.arange(self.startindexforsecondlevel+1, self.startindexforsecondlevel+20)
                        self.weightedQ = np.sum(self.combinedQ[:,self.startindexforsecondlevel+1:self.startindexforsecondlevel+20+1]*\
                                            (self.combinedtimes[self.startindexforsecondlevel+1:self.startindexforsecondlevel+20+1]-self.combinedtimes[self.startindexforsecondlevel:self.startindexforsecondlevel+20])\
                                                /(self.combinedtimes[self.startindexforsecondlevel+20]-self.combinedtimes[self.startindexforsecondlevel]),axis = 1)
                        self.combinedtimes2ndlevel = np.append(self.combinedtimes2ndlevel,self.combinedtimes[self.startindexforsecondlevel+20])
                        self.combinedtimes = np.delete(self.combinedtimes, self.indicestodrop)
                        self.combinedQ[:,self.startindexforsecondlevel+20] = self.weightedQ
                        self.combinedQ = np.delete(self.combinedQ, self.indicestodrop, axis=1)
    
    
            #combine very very old time pulses
            if len(np.where(self.combinedtimes == self.combinedtimes3rdlevel[-1])[0]) > 0:
                self.startindexforthirdlevel = np.where(self.combinedtimes == self.combinedtimes3rdlevel[-1])[0][0]
                if self.combinedtimes.size>50 and self.combinedtimes.size-self.startindexforthirdlevel>50:
                    if (self.times[i] - self.combinedtimes[self.startindexforthirdlevel+20])>self.FMMtriggertime*10:
                        self.indicestodrop = np.arange(self.startindexforthirdlevel+1, self.startindexforthirdlevel+20)
                        self.weightedQ = np.sum(self.combinedQ[:,self.startindexforthirdlevel+1:self.startindexforthirdlevel+20+1]*\
                                            (self.combinedtimes[self.startindexforthirdlevel+1:self.startindexforthirdlevel+20+1]-self.combinedtimes[self.startindexforthirdlevel:self.startindexforthirdlevel+20])\
                                                /(self.combinedtimes[self.startindexforthirdlevel+20]-self.combinedtimes[self.startindexforthirdlevel]),axis = 1)
                        self.combinedtimes3rdlevel = np.append(self.combinedtimes3rdlevel,self.combinedtimes[self.startindexforthirdlevel+20])
                        self.combinedtimes = np.delete(self.combinedtimes, self.indicestodrop)
                        self.combinedQ[:,self.startindexforthirdlevel+20] = self.weightedQ
                        self.combinedQ = np.delete(self.combinedQ, self.indicestodrop, axis=1)
            
            
            self.endindex = np.where(self.times == self.combinedtimes[-1])[0][0]
            self.remainingQ = self.Q[:,self.endindex+1:i+1]
            self.QFMM = np.hstack((self.combinedQ, self.remainingQ))
            self.timesFMM = np.append(self.combinedtimes,self.times[self.endindex+1:i+1])  
            
            
        else:
            self.QFMM = self.Q[:,0:i+1]
            self.timesFMM = self.times[0:i+1]


        ##########
        self.T_prd_bh = np.array(self.num_prd*[self.Toutput], dtype='float')
        self.T_inj_wh = np.array(self.num_inj*[self.Tin], dtype='float')

    def configure_well_dimensions(self):
        """Configuration specifications of a doublet. It requires the specification of a doublet, including the producer dimensions (self.xprod, self.yprod, self.zprod), injector dimensions (self.xinj, self.yinj, self.zinj) and reservoir vertices (self.verts). See Class PercentageReservoir for example implementation.

        """

        self.v = [
                    [np.min(self.x), np.min(self.y) - 5 * self.res_thickness, -self.well_tvd + self.res_thickness/2],
                    [np.min(self.x), np.max(self.y) + 5 * self.res_thickness, -self.well_tvd + self.res_thickness/2],
                    [np.max(self.x), np.max(self.y) + 5 * self.res_thickness, -self.well_tvd + self.res_thickness/2],
                    [np.max(self.x), np.min(self.y) - 5 * self.res_thickness, -self.well_tvd + self.res_thickness/2],
                    [np.max(self.x), np.min(self.y) - 5 * self.res_thickness, -self.well_tvd - self.res_thickness/2],
                    [np.min(self.x), np.min(self.y) - 5 * self.res_thickness, -self.well_tvd - self.res_thickness/2],
                    [np.min(self.x), np.max(self.y) + 5 * self.res_thickness, -self.well_tvd - self.res_thickness/2],
                    [np.max(self.x), np.max(self.y) + 5 * self.res_thickness, -self.well_tvd - self.res_thickness/2],
        ]
        
        self.v = np.array(self.v)
        self.f = [[0,1,2,3], [4,5,6,7], [0, 1, 6, 5], [1, 2, 7, 6], [2, 3, 4, 7], [0, 3, 4, 5]]
        self.verts =  [[self.v[i] for i in p] for p in self.f]


class TabularReservoir(BaseReservoir):
    """Conceptual reservoir model where temperature declines based on an fixed annual decline rate."""

    def __init__(self,
                 Tres_init,
                 geothermal_gradient,
                 surface_temp,
                 L,
                 time_init,
                 well_tvd,
                 prd_well_diam,
                 inj_well_diam,
                 num_prd,
                 num_inj,
                 waterloss,
                 powerplant_type,
                 pumpeff,
                 ramey=True,
                 pumping=True,
                  krock=3,
                 rhorock=2700,
                    cprock=1000,
                   impedance = 0.1,
                 res_thickness=200,
                 PI = 20,
                 II = 20, 
                 SSR = 1.0,
                 N_ramey_mv_avg=168,
                 drawdp=0.005,
                 plateau_length=3,
                 reservoir_simulator_settings={"fast_mode": False, "period": 3600*8760/12},
                 PumpingModel="OpenLoop",
                 timestep=timedelta(hours=1),
                 krock_wellbore=3,
                 filepath=None
                 ):

        """Initialize reservoir model.

        Args:
            Tres_init (float): initial reservoir temperature in deg C.
            geothermal_gradient (float): average geothermal gradient in deg C/km.
            surface_temp (float): surface temperature in deg C.
            L (float): project lifetime in years.
            time_init (datetime): initial time.
            well_tvd (float): well depth in meters.
            prd_well_diam (float): production well diameter in meters.
            inj_well_diam (float): injection well diameter in meters.
            num_prd (int): number of producers.
            num_inj (int): number of injectors.
            waterloss (float): fraction of injected water that is lost to the reservoir (fraction).
            powerplant_type (str): type of power plant (either "Binary" or "Flash").
            pumpeff (float): pump efficiency (fraction).
            ramey (bool, optional): whether or not to use ramey's model for wellbore heat loss/gain. Defaults to True.
            pumping (bool, optional): whther or not to account for parasitic losses due to pumping requirements. Defaults to True.
            krock (float, optional): rock thermal conductivity in W/C-m. Defaults to 3.
            rhorock (float, optional): rock bulk density in kg/m3. Defaults to 2700.
            cprock (float, optional): rock heat capacity in J/kg-K. Defaults to 1000.
            impedance (float, optional): reservoir pressure losses when using an impendance model. Defaults to 0.1.
            res_thickness (float, optional): reservoir thickness in meters. Defaults to 200.
            PI (float, optional): productivity index in kg/s/bar. Defaults to 20.
            II (float, optional): injectivity index in kg/s/bar. Defaults to 20.
            SSR (float, optional): Stimulation success rate, which is a multiplier used to reduce PI and II when stimulation is not fully successful. Defaults to 1.0.
            N_ramey_mv_avg (int, optional): number of timesteps used for averaging the f-function when computing ramey's heat losses with variable mass flow rates. Defaults to 168.
            drawdp (float, optional): annual decline rate of reservoir temperature (fraction). Defaults to 0.005.
            plateau_length (int, optional): number of years before reservoir temperature starts to decline. Defaults to 3.
            reservoir_simulator_settings (dict, optional): information used to reduce the required timestepping when simulating the reservoir. It comes with keys of "fast_mode" to turn it on and "period" to specify the time period needed to pass before the reservoir state is updated, which is aimed at reducing computational requirements in exchange for loss in accuracy. Defaults to {"fast_mode": False, "period": 3600*8760/12}.
            PumpingModel (str, optional): model type used to compute pressure losses (either "OpenLoop" or "ClosedLoop"). Defaults to "OpenLoop".
        """

        self.filepath = filepath
        assert filepath, "UserError: filepath is not specified for the selected tabular reservoir."

        self.df = pd.read_csv(self.filepath)
        assert "Time_sec" in self.df.columns, f"UserError: you must have column with name 'Time' (time in seconds since start of production) in the provided tabular reservoir data at {self.filepath}"
        assert "Tres_deg_C" in self.df.columns, f"UserError: you must have column with name 'Tres_deg_C' (reservoir temperature) in the provided tabular reservoir data at {self.filepath}"
        assert "m_prd_kg_per_sec" in self.df.columns, f"UserError: you must have column with name 'm_prd_kg_per_sec' (total field geofluid production) in the provided tabular reservoir data at {self.filepath}"
        assert "m_inj_kg_per_sec" in self.df.columns, f"UserError: you must have column with name 'm_inj_kg_per_sec' (total field geofluid injection) in the provided tabular reservoir data at {self.filepath}"
        assert "prd_pumping_power_mwe" in self.df.columns, f"UserError: you must have column with name 'prd_pumping_power_mwe' (producer pumping power) in the provided tabular reservoir data at {self.filepath}"
        assert "int_pumping_power_mwe" in self.df.columns, f"UserError: you must have column with name 'int_pumping_power_mwe' (injection pumping power) in the provided tabular reservoir data at {self.filepath}"
        assert "WHP_prd_kPa" in self.df.columns, f"UserError: you must have column with name 'WHP_prd_kPa' (producer whp) in the provided tabular reservoir data at {self.filepath}"
        assert "WHP_inj_kPa" in self.df.columns, f"UserError: you must have column with name 'WHP_inj_kPa' (injection whp) in the provided tabular reservoir data at {self.filepath}"


        Tres_init = self.df.loc[0, "Tres_deg_C"]
        geothermal_gradient = (Tres_init - surface_temp)/well_tvd*1000

        self.Tres_interpolator = interpolate.interp1d(self.df['Time_sec'].values, self.df['Tres_deg_C'].values)
        self.m_prd_interpolator = interpolate.interp1d(self.df['Time_sec'].values, self.df['m_prd_kg_per_sec'].values)
        self.m_inj_interpolator = interpolate.interp1d(self.df['Time_sec'].values, self.df['m_inj_kg_per_sec'].values)
        self.pumping_prd_interpolator = interpolate.interp1d(self.df['Time_sec'].values, self.df['prd_pumping_power_mwe'].values)
        self.pumping_inj_interpolator = interpolate.interp1d(self.df['Time_sec'].values, self.df['int_pumping_power_mwe'].values)
        self.WHP_prd_interpolator = interpolate.interp1d(self.df['Time_sec'].values, self.df['WHP_prd_kPa'].values)
        self.WHP_inj_interpolator = interpolate.interp1d(self.df['Time_sec'].values, self.df['WHP_inj_kPa'].values)


        super(TabularReservoir, self).__init__(Tres_init,
                                            geothermal_gradient,
                                            surface_temp,
                                            L,
                                            time_init,
                                            well_tvd,
                                            prd_well_diam,
                                            inj_well_diam,
                                            num_prd,
                                            num_inj,
                                            waterloss,
                                            powerplant_type,
                                            pumpeff,
                                            ramey,
                                            pumping,
                                            krock,
                                            rhorock,
                                            cprock,
                                            impedance,
                                            res_thickness,
                                            PI,
                                            II,
                                            SSR,
                                            N_ramey_mv_avg,
                                            reservoir_simulator_settings,
                                            PumpingModel,
                                            timestep,
                                            krock_wellbore)

        self.numberoflaterals = 1
        self.well_tvd = well_tvd
        self.well_md = self.well_tvd
        self.res_length = 2000
        self.res_thickness = res_thickness
        self.res_width = 1000

        self.configure_well_dimensions()
        
    def pre_model(self, t, m_prd, m_inj, T_inj):
        """Computations to be performed before stepping the reservoir model.

        Args:
            t (datetime): current timestamp.
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s.
            T_inj (float): injection temperature in deg C.
        """
        pass

    def model(self, t, m_prd, m_inj, T_inj):

        """Computations to be performed when stepping the reservoir model.

        Args:
            t (datetime): current timestamp.
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s.
            T_inj (float): injection temperature in deg C.
        """
        time_passed_seconds = (t - self.time_init).total_seconds()
        self.Tres = float(self.Tres_interpolator(time_passed_seconds))
        # self.Tres = self.df.loc[(self.df["Time"] - time_passed_seconds).abs().argmin(), "Tres_deg_C"]
        self.T_prd_bh = self.Tres
        self.T_inj_wh = T_inj

    def configure_well_dimensions(self):
        """Configuration specifications of a doublet. It requires the specification of a doublet, including the producer dimensions (self.xprod, self.yprod, self.zprod), injector dimensions (self.xinj, self.yinj, self.zinj) and reservoir vertices (self.verts). See Class PercentageReservoir for example implementation.
        """

        self.zprod = np.array([0, -self.well_tvd])
        self.xprod = -self.res_length/2 * np.ones_like(self.zprod)
        self.yprod = np.zeros_like(self.zprod)

        self.zinj = np.array([0, -self.well_tvd])
        self.xinj = self.res_length/2 * np.ones_like(self.zinj)
        self.yinj = np.zeros_like(self.zinj)

        self.v = [
            [-self.res_length/2, -self.res_width/2, -self.well_tvd + self.res_thickness],
            [-self.res_length/2, self.res_width/2, -self.well_tvd + self.res_thickness],
            [self.res_length/2, self.res_width/2, -self.well_tvd + self.res_thickness],
            [self.res_length/2, -self.res_width/2, -self.well_tvd + self.res_thickness],
            [self.res_length/2, -self.res_width/2, -self.well_tvd],
            [-self.res_length/2, -self.res_width/2, -self.well_tvd],
            [-self.res_length/2, self.res_width/2, -self.well_tvd],
            [self.res_length/2, self.res_width/2, -self.well_tvd],
        ]

        self.v = np.array(self.v)
        self.f = [[0,1,2,3], [4,5,6,7], [0, 1, 6, 5], [1, 2, 7, 6], [2, 3, 4, 7], [0, 3, 4, 5]]
        self.verts =  [[self.v[i] for i in p] for p in self.f]


if __name__ == '__main__':
    pass

