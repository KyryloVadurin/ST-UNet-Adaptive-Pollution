import numpy as np
from typing import List, Dict
from .config import SimConfig

class SourceManager:
    """
    Logic for managing stationary (factories) and mobile (traffic) emission sources.
    Handles route generation and emission intensity distribution.
    """
    def __init__(self, config: SimConfig):
        self.config = config
        self.routes = self._generate_routes()
        self.static_sources = self._init_static()
        self.mobile_agents = self._init_mobile()

    def _generate_routes(self):
        """
        Generates linear paths representing road infrastructure across the grid.
        """
        routes = []
        total_r = self.config.num_main_routes + self.config.num_minor_routes
        for _ in range(total_r):
            # Define start and end points for the route
            p1 = (np.random.randint(0, self.config.grid_x), np.random.randint(0, self.config.grid_y))
            p2 = (np.random.randint(0, self.config.grid_x), np.random.randint(0, self.config.grid_y))
            
            # Interpolate coordinates to create a continuous path
            length = max(self.config.grid_x, self.config.grid_y) * 2
            x = np.linspace(p1[0], p2[0], length).astype(int)
            y = np.linspace(p1[1], p2[1], length).astype(int)
            routes.append(np.vstack((x, y)).T)
        return routes

    def _init_static(self):
        """
        Initializes stationary sources with fixed coordinates and randomized emission power.
        """
        # Coordinate generation with safety margin from boundaries
        x = np.random.randint(2, self.config.grid_x-2, self.config.num_static_sources)
        y = np.random.randint(2, self.config.grid_y-2, self.config.num_static_sources)
        
        # Intensity range application
        low, high = self.config.static_intensity_range
        intensities = np.random.uniform(low, high, self.config.num_static_sources)
        return np.column_stack((x, y, intensities))

    def _init_mobile(self):
        """
        Initializes individual moving agents assigned to pre-generated routes.
        """
        agents = []
        low, high = self.config.mobile_intensity_range
        for _ in range(self.config.num_mobile_sources):
            # Assign agent to a random route and starting position
            r_idx = np.random.randint(0, len(self.routes))
            agents.append({
                'route_idx': r_idx,
                'pos_idx': np.random.randint(0, len(self.routes[r_idx])),
                'speed': np.random.randint(1, 3),
                'intensity': np.random.uniform(low, high)
            })
        return agents

    def get_emissions_grid(self):
        """
        Rasterizes all active emission sources onto the 2D grid for the current step.
        """
        grid = np.zeros((self.config.grid_x, self.config.grid_y))
        
        # Add contributions from stationary sources
        for sx, sy, val in self.static_sources:
            grid[int(sx), int(sy)] += val
            
        # Update mobile positions and add their emission contributions
        for a in self.mobile_agents:
            route = self.routes[a['route_idx']]
            # Move agent along the route with wrap-around logic
            a['pos_idx'] = (a['pos_idx'] + a['speed']) % len(route)
            x, y = route[a['pos_idx']]
            grid[x, y] += a['intensity']
            
        return grid