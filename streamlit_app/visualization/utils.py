"""Utility functions for visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
import seaborn as sns

# Color palette for plots
colors = 24 * sns.color_palette()
linestyles = ['-', '--', '-.', ':']


def plot_cols(dfs: Dict[str, pd.DataFrame],
              span: Union[range, list],
              quantities: List[str], 
              figsize: tuple = (10, 10),
              xlabel: str = "World Time",
              ylabels: Optional[List[str]] = None,
              ylogscale: Union[bool, List[bool]] = False,
              use_title: bool = True,
              legend_loc: Union[bool, str] = "lower right",
              manual_legends: Optional[List[str]] = None,
              color_per_col: bool = True,
              use_linestyles: bool = True,
              blackout_first: bool = False,
              formattime: bool = False,
              grid: bool = True,
              dpi: int = 100):
    """Visualize columns of a dataframe.

    Args:
        dfs: Dictionary of dataframes to plot with key as names.
        span: Range of indices to plot.
        quantities: Dataframe columns to plot.
        figsize: Figure size. Defaults to (10,10).
        xlabel: Label of the x axis. Defaults to "World Time".
        ylabels: Labels of the y axes. Defaults to None.
        ylogscale: Whether or not to use log-scale on the y axis. Defaults to False.
        use_title: Whether or not to use a figure title. Defaults to True.
        legend_loc: Legend location. Defaults to "lower right".
        manual_legends: List to override default legends. Defaults to None.
        color_per_col: Whether or not to assign one color to each column. Defaults to True.
        use_linestyles: Whether or not to use linestyles. If False, all quantities are plotted with solid linestyle. Defaults to True.
        blackout_first: Whether or not to plot the very first quantity in black color. Defaults to False.
        formattime: Whether or not to apply time formatting. Defaults to False.
        dpi: Figure dpi resolution. Defaults to 100.

    Returns:
        matplotlib.figure.Figure: figure object
    """
    if len(quantities) == 1:
        fig, axes = plt.subplots(1, 1, figsize=figsize, sharex=True, dpi=dpi)
        axes = [axes]
    else:
        fig, axes = plt.subplots(len(quantities), 1, figsize=figsize, sharex=True, dpi=dpi)

    df_plots = [df.iloc[span].copy() for df in dfs.values()]
    counter = 0
    
    for k, df_plot in enumerate(df_plots):
        for i, q in enumerate(quantities):
            q = q if isinstance(q, list) else [q]
            for col in q:
                if all([qi in df_plot.columns for qi in q]):
                    axes[i].plot(df_plot.index, df_plot[col], 
                               color=colors[counter], 
                               linestyle=linestyles[k if use_linestyles else 0])
                if color_per_col:
                    counter += 1
                if legend_loc:
                    axes[i].legend(q, loc=legend_loc)
            
            axes[i].yaxis.tick_right()

            if grid:
                axes[i].grid(color='lightgray')

        if not color_per_col:
            counter += 1
        else:
            counter = 0
    
    if use_title:
        if len(dfs.keys()) > 1:
            axes[0].set_title("\n".join([f"{k}: {linestyles[i]}" for i, k in enumerate(dfs.keys())]))
        else:
            axes[0].set_title(list(dfs.keys())[0])
    
    axes[-1].set_xlabel(xlabel)
    if formattime:
        if "time" in xlabel.lower() or "date" in xlabel.lower():
            plt.gcf().autofmt_xdate()
    
    if ylabels:
        for i, ylabel in enumerate(ylabels):
            axes[i].set_ylabel(ylabel)
    if ylogscale:
        ylogscale_list = ylogscale if isinstance(ylogscale, list) else [ylogscale] * len(quantities)
        for i, log in enumerate(ylogscale_list):
            if log:
                axes[i].set_yscale("log")
    if manual_legends:
        axes[0].legend(list(dfs.keys()), loc='upper right', fontsize=10)
    if blackout_first:
        axes[0].plot(df_plot.index, df_plot[quantities[0]], color='black')

    return fig


def plot_ex(worlds: Dict[str, Dict[str, float]],
            figsize: tuple = (10, 10),
            dpi: int = 150,
            colors: List = colors,
            fontsize: int = 10):
    """Visualize expenditure of multiple projects. 

    Args:
        worlds: Dictionary of projects or worlds to visualize, where each value is a dict of expenditures.
        figsize: Figure size. Defaults to (10,10).
        dpi: Figure dpi resolution. Defaults to 150.
        colors: List of colors to be used. Defaults to seaborn.color_palette().
        fontsize: Font size. Defaults to 10.

    Returns:
        matplotlib.figure.Figure: figure object
    """
    fig, axes = plt.subplots(1, len(worlds), figsize=figsize, dpi=dpi)
    axes = [axes] if len(worlds) == 1 else axes

    def func(pct, allvals):
        absolute = int(np.round(pct/100.*np.sum(allvals)))
        return "{:.0f}%".format(pct)
    
    for i, (title, present_per_unit) in enumerate(worlds.items()):
        ex = [v for _,v in present_per_unit.items() if v > 0]
        labels = [k for k,v in present_per_unit.items() if v > 0]
        wedges, _, _ = axes[i].pie(x=ex,
                                   pctdistance=0.8,
                                   labels=labels, 
                                   colors=colors, 
                                   autopct=lambda pct: func(pct, ex),
                                   textprops=dict(color="w", weight="bold", fontsize=fontsize))

        axes[i].legend(wedges, labels,
                title=title,
                loc="center left",
        )

    return fig

