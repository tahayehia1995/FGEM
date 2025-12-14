"""Economics visualization functions."""

import matplotlib.pyplot as plt
import numpy as np
import warnings
from typing import Optional, List
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

colors = 24 * sns.color_palette()


class EconomicsVisualizer:
    """Visualizer for economics-related plots."""
    
    @staticmethod
    def plot_capex_opex_matplotlib(world, 
                                   figsize: tuple = (10, 10),
                                   dpi: int = 150, 
                                   fontsize: int = 10, 
                                   colors: list = colors):
        """Plot CAPEX and OPEX as pie charts using matplotlib.

        Args:
            world: World instance with computed economics.
            figsize: Figure size. Defaults to (10, 10).
            dpi: Resolution dpi. Defaults to 150.
            fontsize: Font size. Defaults to 10.
            colors: Color palette. Defaults to seaborn colors.

        Returns:
            matplotlib.figure.Figure: figure
        """
        if not hasattr(world, "present_capex_per_unit"):
            warnings.warn("Warning: economics are computed based on the latest simulation timestep.")
            world.compute_economics()

        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

        def func(pct, allvals):
            absolute = int(np.round(pct/100.*np.sum(allvals)))
            if pct < 1:
                return ""
            return "{:.1f}%".format(pct)

        expenditures = {}

        include = ['Power Plant', 'Interconnection', 'Exploration', 
                'Drilling', 'Stimulation', 'Gathering System', 'Pumps', 'TES', 'Battery']
        costs = {k:v for k,v in world.present_capex_per_unit.items()}
        costs["Pumps"] = costs.get("Production Pumps", 0) + costs.get("Injection Pumps", 0)
        costs_final = {k:costs.get(k, 0) for k in include}
        expenditures["CAPEX"] = costs_final

        include = ['Power Plant', 'Wellsite', 'Makeup Water']
        costs = {k:v for k,v in world.present_opex_per_unit.items()}
        costs_final = {k:costs.get(k, 0) for k in include}
        expenditures["OPEX"] = costs_final

        for i, (title, present_per_unit) in enumerate(expenditures.items()):
            ex, labels = [], []
            for k,v in present_per_unit.items():
                if v != 0: 
                    ex.append(np.empty([]) if (v < 1 and title=='CAPEX') else v )
                    labels.append(k)
            wedges, _, _ = axes[i].pie(x=ex,
                                    pctdistance=0.8,
                                    colors=colors, 
                                    autopct=lambda pct: func(pct, ex),
                                    textprops=dict(color="w", weight="bold", fontsize=fontsize))

            axes[i].legend(wedges, labels,
                        fontsize=10,
                    title=title,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.1, 0.0, 1.2),
                        ncols=2)

        plt.tight_layout()

        return fig
    
    @staticmethod
    def plot_capex_opex_plotly(world):
        """Plot CAPEX and OPEX as pie charts using Plotly.

        Args:
            world: World instance with computed economics.

        Returns:
            plotly.graph_objects.Figure: figure
        """
        if not hasattr(world, "present_capex_per_unit"):
            warnings.warn("Warning: economics are computed based on the latest simulation timestep.")
            world.compute_economics()

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "pie"}]],
            subplot_titles=("CAPEX", "OPEX")
        )

        # CAPEX
        include_capex = ['Power Plant', 'Interconnection', 'Exploration', 
                        'Drilling', 'Stimulation', 'Gathering System', 'Pumps', 'TES', 'Battery']
        costs = {k:v for k,v in world.present_capex_per_unit.items()}
        costs["Pumps"] = costs.get("Production Pumps", 0) + costs.get("Injection Pumps", 0)
        capex_data = {k: costs.get(k, 0) for k in include_capex if costs.get(k, 0) > 0}
        
        if capex_data:
            fig.add_trace(
                go.Pie(labels=list(capex_data.keys()), 
                      values=list(capex_data.values()),
                      name="CAPEX",
                      textinfo="label+percent"),
                row=1, col=1
            )

        # OPEX
        include_opex = ['Power Plant', 'Wellsite', 'Makeup Water']
        costs = {k:v for k,v in world.present_opex_per_unit.items()}
        opex_data = {k: costs.get(k, 0) for k in include_opex if costs.get(k, 0) > 0}
        
        if opex_data:
            fig.add_trace(
                go.Pie(labels=list(opex_data.keys()), 
                      values=list(opex_data.values()),
                      name="OPEX",
                      textinfo="label+percent"),
                row=1, col=2
            )

        fig.update_layout(
            title_text="Project Economics",
            showlegend=True,
            height=500
        )

        return fig
    
    @staticmethod
    def plot_price_distribution_matplotlib(world, figsize: tuple = (8, 5), dpi: int = 100):
        """Plot power wholesale market price distribution using matplotlib.

        Args:
            world: World instance with market data.
            figsize: Figure size. Defaults to (8, 5).
            dpi: Resolution dpi. Defaults to 100.

        Returns:
            matplotlib.figure.Figure: figure
        """
        fig = plt.figure(figsize=figsize, dpi=dpi)
        world.df_market.price.hist(histtype='step', bins=200)
        plt.xlim([world.df_market.price.min(), world.df_market.price.max()])
        plt.xlabel("Price [$/MWh]")
        plt.ylabel("Frequency")
        plt.title("Wholesale Market Price Distribution")
        return fig
    
    @staticmethod
    def plot_price_distribution_plotly(world):
        """Plot power wholesale market price distribution using Plotly.

        Args:
            world: World instance with market data.

        Returns:
            plotly.graph_objects.Figure: figure
        """
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=world.df_market.price,
            nbinsx=200,
            name="Price Distribution",
            marker_color='lightblue'
        ))
        fig.update_layout(
            title="Wholesale Market Price Distribution",
            xaxis_title="Price [$/MWh]",
            yaxis_title="Frequency",
            height=400
        )
        return fig
    
    @staticmethod
    def plot_cashflow_multi_discount_plotly(world, discount_rates: List[float]):
        """Plot cumulative discounted cashflow for multiple discount rates with payback markers.
        
        Args:
            world: World instance with computed economics.
            discount_rates: List of discount rates (as decimals, e.g., [0.03, 0.05, 0.07]).
            
        Returns:
            plotly.graph_objects.Figure: figure
        """
        if not hasattr(world, "df_annual_nominal"):
            warnings.warn("Warning: economics are computed based on the latest simulation timestep.")
            world.compute_economics()
        
        if "Cashflow [$MM]" not in world.df_annual_nominal.columns:
            world.compute_economics()
        
        years = world.df_annual_nominal.index.values
        cashflow_nominal = world.df_annual_nominal["Cashflow [$MM]"].values
        
        fig = go.Figure()
        
        # Color palette for different discount rates
        import plotly.colors as pc
        n_rates = len(discount_rates)
        if n_rates <= 10:
            color_sequence = pc.qualitative.Set3[:n_rates]
        else:
            color_sequence = pc.sample_colorscale("Viridis", [i/(n_rates-1) for i in range(n_rates)])
        
        payback_years = []
        
        for i, d in enumerate(sorted(discount_rates)):
            # Discount annual cashflows
            discount_factors = np.power(1.0 + d, years - years[0])
            cashflow_discounted = cashflow_nominal / discount_factors
            cumulative = np.cumsum(cashflow_discounted)
            
            # Find payback year (first year where cumulative >= 0)
            positive_mask = cumulative >= 0
            if np.any(positive_mask):
                # argmax on boolean array returns index of first True
                payback_idx = int(np.argmax(positive_mask))
                payback_year = float(years[payback_idx])
            else:
                payback_year = None
            
            payback_years.append(payback_year)
            
            label = f"d = {d*100:.1f}%"
            if payback_year is not None:
                label += f" (PBP: {int(payback_year - years[0])} yr)"
            else:
                label += " (no payback)"
            
            fig.add_trace(go.Scatter(
                x=years,
                y=cumulative,
                mode='lines+markers',
                name=label,
                line=dict(width=2.5, color=color_sequence[i]),
                marker=dict(size=6),
            ))
            
            # Add payback marker (vertical line at payback year)
            if payback_year is not None:
                payback_cumulative = cumulative[payback_idx]
                y_min = float(np.nanmin(cumulative))
                y_max = float(np.nanmax(cumulative))
                y_range = y_max - y_min if y_max > y_min else abs(y_max) if y_max != 0 else 1.0
                fig.add_trace(go.Scatter(
                    x=[payback_year, payback_year],
                    y=[y_min - 0.05 * y_range, y_max + 0.05 * y_range],
                    mode='lines',
                    name=f"Payback d={d*100:.1f}%",
                    line=dict(dash='dash', width=1.5, color=color_sequence[i]),
                    showlegend=False,
                    hoverinfo='skip',
                ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1, 
                     annotation_text="Zero cashflow", annotation_position="right")
        
        fig.update_layout(
            title="<b>Cumulative Discounted Cashflow at Different Discount Rates</b>",
            xaxis_title="<b>Year</b>",
            yaxis_title="<b>Cumulative Discounted Cashflow [$MM]</b>",
            height=550,
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=60, r=40, t=60, b=50),
        )
        
        return fig
    
    @staticmethod
    def plot_cashflow_multi_discount_matplotlib(world, discount_rates: List[float], 
                                                figsize: tuple = (10, 6), dpi: int = 100):
        """Plot cumulative discounted cashflow for multiple discount rates with payback markers.
        
        Args:
            world: World instance with computed economics.
            discount_rates: List of discount rates (as decimals, e.g., [0.03, 0.05, 0.07]).
            figsize: Figure size. Defaults to (10, 6).
            dpi: Resolution dpi. Defaults to 100.
            
        Returns:
            matplotlib.figure.Figure: figure
        """
        if not hasattr(world, "df_annual_nominal"):
            warnings.warn("Warning: economics are computed based on the latest simulation timestep.")
            world.compute_economics()
        
        if "Cashflow [$MM]" not in world.df_annual_nominal.columns:
            world.compute_economics()
        
        years = world.df_annual_nominal.index.values
        cashflow_nominal = world.df_annual_nominal["Cashflow [$MM]"].values
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        n_rates = len(discount_rates)
        color_cycle = plt.cm.tab10(np.linspace(0, 1, min(n_rates, 10)))
        
        for i, d in enumerate(sorted(discount_rates)):
            # Discount annual cashflows
            discount_factors = np.power(1.0 + d, years - years[0])
            cashflow_discounted = cashflow_nominal / discount_factors
            cumulative = np.cumsum(cashflow_discounted)
            
            # Find payback year
            positive_mask = cumulative >= 0
            if np.any(positive_mask):
                # argmax on boolean array returns index of first True
                payback_idx = int(np.argmax(positive_mask))
                payback_year = float(years[payback_idx])
                payback_cumulative = float(cumulative[payback_idx])
            else:
                payback_year = None
                payback_cumulative = None
            
            color = color_cycle[i % len(color_cycle)]
            label = f"d = {d*100:.1f}%"
            if payback_year is not None:
                label += f" (PBP: {int(payback_year - years[0])} yr)"
            else:
                label += " (no payback)"
            
            ax.plot(years, cumulative, marker='o', linewidth=2.5, markersize=6, 
                   label=label, color=color)
            
            # Add payback marker
            if payback_year is not None:
                ax.axvline(x=payback_year, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
                ax.plot(payback_year, payback_cumulative, marker='*', markersize=12, 
                       color=color, markeredgecolor='k', markeredgewidth=0.5)
        
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.set_xlabel("Year", fontweight="bold")
        ax.set_ylabel("Cumulative Discounted Cashflow [$MM]", fontweight="bold")
        ax.set_title("Cumulative Discounted Cashflow at Different Discount Rates", fontweight="bold")
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        return fig

