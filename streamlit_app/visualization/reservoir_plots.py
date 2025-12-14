"""Reservoir visualization functions."""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import plotly.graph_objects as go


class ReservoirVisualizer:
    """Visualizer for reservoir-related plots."""
    
    @staticmethod
    def plot_doublet_matplotlib(reservoir, dpi: int = 150):
        """Visualize a doublet of the proposed system using matplotlib.

        Args:
            reservoir: Reservoir instance with configured well dimensions.
            dpi: Figure dpi resolution. Defaults to 150.

        Returns:
            matplotlib.figure.Figure: figure
        """
        assert hasattr(reservoir, "zprod"), \
            "Implementation Error: You must define the wellbore and reservoir dimensions to plot doublets! Define method subsurface.BaseReservoir.configure_well_dimensions"

        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')

        ax.add_collection3d(Poly3DCollection(reservoir.verts, alpha=0.1, color="tab:orange"))
        ax.plot(reservoir.xinj, reservoir.yinj, reservoir.zinj, 'tab:blue', linewidth=4)
        ax.plot(reservoir.xprod, reservoir.yprod, reservoir.zprod, 'tab:red', linewidth=4)

        ax.set_xlim([np.min(reservoir.v[:,0]) - 200, np.max(reservoir.v[:,0]) + 200])
        ax.set_ylim([np.min(reservoir.v[:,1]) - 200, np.max(reservoir.v[:,1]) + 200])
        ax.set_zlim([np.min(reservoir.v[:,2]) - 500, 0])

        col1_patch = mpatches.Patch(color="tab:orange", label='Reservoir')
        col2_patch = mpatches.Patch(color="tab:blue", label='Injector')
        col3_patch = mpatches.Patch(color="tab:red", label='Producer')
        handles = [col1_patch, col2_patch, col3_patch]

        if hasattr(reservoir, "zlat"):
            for j in range(reservoir.xlat.shape[1]):
                ax.plot(reservoir.xlat[:,j], reservoir.ylat[:,j], reservoir.zlat[:,j],
                        linewidth=2, color="black")

            col4_patch = mpatches.Patch(color="black", label='Laterals')
            handles.append(col4_patch)
            
        plt.legend(handles=handles)

        return fig
    
    @staticmethod
    def plot_doublet_plotly(reservoir):
        """Visualize a doublet of the proposed system using Plotly.

        Args:
            reservoir: Reservoir instance with configured well dimensions.

        Returns:
            plotly.graph_objects.Figure: figure
        """
        assert hasattr(reservoir, "zprod"), \
            "Implementation Error: You must define the wellbore and reservoir dimensions to plot doublets!"

        fig = go.Figure()

        # Add reservoir (as a box)
        if hasattr(reservoir, 'verts'):
            for vert in reservoir.verts:
                x_coords = [v[0] for v in vert]
                y_coords = [v[1] for v in vert]
                z_coords = [v[2] for v in vert]
                fig.add_trace(go.Mesh3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    opacity=0.3,
                    color='orange',
                    name='Reservoir'
                ))

        # Add injector well
        fig.add_trace(go.Scatter3d(
            x=reservoir.xinj,
            y=reservoir.yinj,
            z=reservoir.zinj,
            mode='lines',
            line=dict(color='blue', width=8),
            name='Injector'
        ))

        # Add producer well
        fig.add_trace(go.Scatter3d(
            x=reservoir.xprod,
            y=reservoir.yprod,
            z=reservoir.zprod,
            mode='lines',
            line=dict(color='red', width=8),
            name='Producer'
        ))

        # Add laterals if they exist
        if hasattr(reservoir, "zlat"):
            for j in range(reservoir.xlat.shape[1]):
                fig.add_trace(go.Scatter3d(
                    x=reservoir.xlat[:,j],
                    y=reservoir.ylat[:,j],
                    z=reservoir.zlat[:,j],
                    mode='lines',
                    line=dict(color='black', width=4),
                    name='Laterals' if j == 0 else '',
                    showlegend=(j == 0)
                ))

        fig.update_layout(
            title="Reservoir Doublet Visualization",
            scene=dict(
                xaxis_title="X [m]",
                yaxis_title="Y [m]",
                zaxis_title="Z [m]",
                aspectmode='data'
            ),
            height=600
        )

        return fig

