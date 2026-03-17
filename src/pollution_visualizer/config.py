from dataclasses import dataclass

# Data container for visualization styling and export parameters
@dataclass
class VisConfig:
    # Color mapping and aesthetic properties
    cmap_main: str = "magma"
    cmap_div: str = "RdBu_r"
    sensor_color: str = "#00ffff"
    
    # Rendering and resolution parameters
    dpi: int = 120
    
    # Figure geometry definitions
    figsize_std: tuple = (10, 6)
    figsize_square: tuple = (8, 8)
    figsize_wide: tuple = (18, 5)
    
    # Typography settings
    title_fontsize: int = 14