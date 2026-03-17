import h5py
import numpy as np
import torch
import json
import random
from torch.utils.data import IterableDataset, get_worker_info
from .transforms import PollutionTransforms
from .augmentations import SpatialAugmentations

class PollutionStreamingDataset(IterableDataset):
    """
    Streaming dataset implementation using HDF5 for memory-efficient training.
    Supports multi-worker loading and spatial augmentations.
    """
    def __init__(self, config, h5_path, transforms: PollutionTransforms, mode: str = 'train'):
        self.h5_path = h5_path
        self.transforms = transforms
        self.use_wind = config.use_wind
        self.grid_size = config.grid_size
        self.samples = []
        self.aug = SpatialAugmentations(config.grid_size) if (mode == 'train' and config.use_augmentations) else None

        # Data partitioning logic based on scenarios
        with h5py.File(self.h5_path, 'r') as f:
            scens = sorted([k for k in f.keys() if isinstance(f[k], h5py.Group)])
            n = len(scens)
            rng = random.Random(config.random_seed)
            rng.shuffle(scens)
            
            t_idx = int(n * config.test_split)
            v_idx = int(n * (config.test_split + config.val_split))
            
            if mode == 'test': 
                active = scens[:t_idx]
            elif mode == 'val': 
                active = scens[t_idx:v_idx]
            else: 
                active = scens[v_idx:]
                
            # Mapping scenario groups and layout indices to samples
            for sn in active:
                for li in range(f[sn]['sensor_readings'].shape[0]):
                    self.samples.append((sn, li))
                    
        print(f"Dataset {mode}: {len(self.samples)} samples.")

    def __len__(self): 
        return len(self.samples)

    def __iter__(self):
        """
        Implementation of the iterator with multi-process worker distribution.
        """
        worker_info = get_worker_info()
        my_samples = self.samples
        
        # Slicing samples for parallel processing
        if worker_info is not None:
            per = int(np.ceil(len(self.samples) / float(worker_info.num_workers)))
            my_samples = self.samples[worker_info.id * per : (worker_info.id + 1) * per]
            
        # Data generation loop
        with h5py.File(self.h5_path, 'r', libver='latest', swmr=True) as f:
            for sn, li in my_samples:
                g = f[sn]
                meta = json.loads(g.attrs.get('config', '{}'))
                wind = np.array(meta.get('scenario_initial_wind', [0.0, 0.0]))
                
                # Reading data from HDF5
                r = g['sensor_readings'][li].copy()
                c = g['sensor_coords'][li].copy()
                target = g['ground_truth'][:].copy()
                
                # Applying spatial augmentations if active
                if self.aug: 
                    r, c, wind, target = self.aug(r, c, wind, target)
                
                # Yielding transformed tensors
                yield self.transforms.transform(r, c, wind, target, use_wind=self.use_wind)