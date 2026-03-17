import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from .config import VisConfig

class AnimationPlotter:
    """
    Handles the generation of animated visualizations for pollution dynamics.
    """
    def __init__(self, config: VisConfig):
        self.cfg = config

    def create_gif(self, gt_history, output_path="pollution_anim.gif", fps=10):
        """
        Creates an animation of pollution dispersion and saves it as a GIF file.
        """
        # Matplotlib figure and axis initialization
        fig, ax = plt.subplots(figsize=(7, 7), dpi=self.cfg.dpi)

        # Initial frame setup
        im = ax.imshow(gt_history[0], cmap=self.cfg.cmap_main, origin='lower', 
                       animated=True, vmin=np.min(gt_history), vmax=np.max(gt_history))
        plt.colorbar(im, ax=ax, label="Concentration")
        title = ax.set_title("Pollution Dynamics: Step 0")

        # Animation update logic for each frame
        def update(frame):
            im.set_array(gt_history[frame])
            title.set_text(f"Pollution Dynamics: Step {frame}")
            return [im, title]

        # FuncAnimation object creation
        anim = FuncAnimation(fig, update, frames=len(gt_history), interval=1000/fps, blit=True)

        # Export logic using PillowWriter
        print(f"Saving animation to {output_path}...")
        anim.save(output_path, writer=PillowWriter(fps=fps))
        plt.close()
        print("Animation saved successfully.")