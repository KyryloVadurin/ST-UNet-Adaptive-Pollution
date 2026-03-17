import matplotlib.pyplot as plt
import numpy as np

def plot_results(y_true, y_pred, coords, metrics, title="Model Evaluation"):
    """
    Visualization of the real map vs. prediction comparison.
    Station coordinates are displayed on both plots.
    """
    # Figure and axes initialization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16)

    # Stylization parameters for sensor markers
    scatter_kwargs = {
        'c': 'cyan', 
        'marker': '^', 
        's': 35, 
        'label': 'Sensors', 
        'edgecolors': 'black', 
        'linewidths': 0.5
    }

    # 1. Ground Truth visualization logic (Left Panel)
    im0 = axes[0].imshow(y_true, cmap='hot', origin='lower', vmin=0, vmax=1.0)
    axes[0].scatter(coords[:, 1], coords[:, 0], **scatter_kwargs)
    axes[0].set_title("Ground Truth", fontsize=12)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Prediction visualization logic (Right Panel)
    im1 = axes[1].imshow(y_pred, cmap='hot', origin='lower', vmin=0, vmax=1.0)
    axes[1].scatter(coords[:, 1], coords[:, 0], **scatter_kwargs)
    axes[1].set_title("Prediction", fontsize=12)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Layout adjustment and rendering
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_uncertainty(y_true, mean_pred, uncertainty, coords):
    """
    Visualization of Stochastic Inference results (Monte Carlo Dropout).
    Station coordinates are displayed across all diagnostic maps.
    """
    # Multi-panel figure initialization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    scatter_kwargs = {'c': 'cyan', 'marker': '^', 's': 30, 'edgecolors': 'black'}

    # Metadata and plotting configuration
    titles = ["Ground Truth", "Mean Prediction", "Uncertainty (Variance)"]
    data = [y_true, mean_pred, uncertainty]
    cmaps = ['hot', 'hot', 'viridis']

    # Iterative sub-plot generation
    for i in range(3):
        im = axes[i].imshow(data[i], cmap=cmaps[i], origin='lower')
        axes[i].scatter(coords[:, 1], coords[:, 0], **scatter_kwargs)
        axes[i].set_title(titles[i])
        plt.colorbar(im, ax=axes[i])

    # Final layout formatting
    plt.tight_layout()
    plt.show()