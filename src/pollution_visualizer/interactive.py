import plotly.graph_objects as go
import numpy as np
from .config import VisConfig

class InteractivePlotter:
    """
    Handles advanced 3D and interactive visualizations using the Plotly engine.
    """
    def __init__(self, config: VisConfig):
        self.cfg = config

    def plot_3d_interactive(self, gt_step, title="Interactive 3D Map"):
        """
        Renders a rotatable 3D surface plot of the pollution concentration.
        """
        z_data = gt_step
        
        # Surface data definition
        fig = go.Figure(data=[go.Surface(z=z_data, colorscale='Magma')])
        
        # Layout and axis labeling
        fig.update_layout(title=title, width=800, height=800,
                          scene=dict(zaxis_title="Concentration", xaxis_title="X", yaxis_title="Y"))
        fig.show()

    def plot_space_time_cube(self, gt_history, skip_steps=2):
        """
        Visualizes the temporal evolution of spatial maps as a vertical stack of planes.
        """
        subset = gt_history[::skip_steps]
        time_steps, x_size, y_size = subset.shape
        
        fig = go.Figure()
        
        # Loop through time steps to generate stacked spatial layers
        for t in range(time_steps):
            # Calculate the vertical offset for the current time slice
            z_offset = np.full((x_size, y_size), t * 4) 
            
            # Add trace for the specific time layer
            fig.add_trace(go.Surface(
                z=z_offset, x=np.arange(y_size), y=np.arange(x_size),
                surfacecolor=subset[t], colorscale='Magma', showscale=False, opacity=0.5
            ))
            
        # Final visualization parameters
        fig.update_layout(title="Space-Time Cube (Vertical Axis is Time)", width=900, height=800)
        fig.show()