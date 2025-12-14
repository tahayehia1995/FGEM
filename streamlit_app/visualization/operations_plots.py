"""Operations visualization functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, Union, List
from .utils import plot_cols


class OperationsVisualizer:
    """Visualizer for operations-related plots."""
    
    @staticmethod
    def plot_operations_matplotlib(world,
                                   span: Optional[Union[range, list]] = None,
                                   qdict: Optional[Dict[str, str]] = None,
                                   figsize: tuple = (10, 12),
                                   legend_loc: Union[bool, str] = False,
                                   dpi: int = 100,
                                   formattime: bool = False):
        """Plot operational parameters using matplotlib.

        Args:
            world: World instance with simulation results.
            span: Range of timesteps to plot. Defaults to None.
            qdict: Dictionary mapping column names to y-axis labels.
            figsize: Figure size. Defaults to (10,12).
            legend_loc: Legend location. Defaults to False.
            dpi: Resolution dpi. Defaults to 100.
            formattime: Whether to format time axis. Defaults to False.

        Returns:
            matplotlib.figure.Figure: figure
        """
        if not hasattr(world, "df_records"):
            warnings.warn("Warning: economics are computed based on the latest simulation timestep.")
            world.compute_economics()

        if qdict is None:
            qdict = {
                "LMP [$/MWh]": "Electricity Price \n [$/MWh]",
                "Atm Temp [deg C]": "Ambient Temp. \n [° C]",
                "Res Temp [deg C]": "Reservoir Temp. \n [° C]",
                'Inj Temp [deg C]': "Injector Temp. \n [° C]",
                "Net Power Output [MWe]": "Net Power Output \n [MWe]",
                'M_Produced [kg/s]': "Field Production \n [kg/s]",
                "Pumping Power [MWe]": "Pumping Power \n [MWe]"
            }

        quantities = list(qdict.keys())
        ylabels = list(qdict.values())

        span = span if span else range(int(0.01*world.max_simulation_steps), world.step_idx)
        fig = plot_cols({" ": world.df_records}, span, quantities, 
                       figsize=figsize, ylabels=ylabels, legend_loc=legend_loc, 
                       dpi=dpi, formattime=formattime)
        
        return fig
    
    @staticmethod
    def plot_operations_plotly(world,
                               span: Optional[Union[range, list]] = None,
                               qdict: Optional[Dict[str, str]] = None):
        """Plot operational parameters using Plotly.

        Args:
            world: World instance with simulation results.
            span: Range of timesteps to plot. Defaults to None.
            qdict: Dictionary mapping column names to y-axis labels.

        Returns:
            plotly.graph_objects.Figure: figure
        """
        if not hasattr(world, "df_records"):
            warnings.warn("Warning: economics are computed based on the latest simulation timestep.")
            world.compute_economics()

        if qdict is None:
            qdict = {
                "LMP [$/MWh]": "Electricity Price [$/MWh]",
                "Atm Temp [deg C]": "Ambient Temp. [°C]",
                "Res Temp [deg C]": "Reservoir Temp. [°C]",
                'Inj Temp [deg C]': "Injector Temp. [°C]",
                "Net Power Output [MWe]": "Net Power Output [MWe]",
                'M_Produced [kg/s]': "Field Production [kg/s]",
                "Pumping Power [MWe]": "Pumping Power [MWe]"
            }

        quantities = list(qdict.keys())
        ylabels = list(qdict.values())
        
        span = span if span else range(int(0.01*world.max_simulation_steps), world.step_idx)
        df_plot = world.df_records.iloc[span].copy()
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=len(quantities), cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=ylabels
        )
        
        for i, (col, ylabel) in enumerate(zip(quantities, ylabels)):
            if col in df_plot.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_plot.index,
                        y=df_plot[col],
                        mode='lines',
                        name=ylabel,
                        showlegend=False
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            height=200 * len(quantities),
            title_text="Operational Parameters",
            xaxis_title="Time"
        )
        
        return fig

