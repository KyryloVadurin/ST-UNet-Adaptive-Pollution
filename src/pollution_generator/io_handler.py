import h5py
import json
import os
import numpy as np
from tqdm import tqdm
from .config import SimConfig
from .engine import Simulator

def generate_dataset_h5(config: SimConfig, num_scenarios: int, filename: str):
    """
    Core dataset generation logic using HDF5 format for efficient storage.
    """
    # Directory verification and creation logic
    if os.path.dirname(filename) and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Initialize HDF5 file write process
    with h5py.File(filename, 'w') as f:
        # Store global configuration as file attribute
        f.attrs['global_config'] = json.dumps(config.to_dict())

        # Main generation loop for individual scenarios
        for i in tqdm(range(num_scenarios), desc=f"Generating {os.path.basename(filename)}"):
            # Execute simulation engine for a single scenario instance
            sim = Simulator(config)
            gt, sensors, layouts, win_vec = sim.run_scenario()

            # Create scenario group and store compressed datasets
            group = f.create_group(f"scenario_{i:04d}")
            group.create_dataset("ground_truth", data=gt, compression="gzip", compression_opts=4)
            group.create_dataset("sensor_readings", data=sensors, compression="gzip", compression_opts=4)
            group.create_dataset("sensor_coords", data=np.array(layouts))

            # Serialize scenario-specific metadata
            meta = config.to_dict()
            meta['scenario_initial_wind'] = [float(win_vec[0]), float(win_vec[1])]
            group.attrs['config'] = json.dumps(meta)

    # Final status report
    file_size_mb = os.path.getsize(filename) / 1e6
    print(f"Dataset saved to {filename}. Size: {file_size_mb:.2f} MB")