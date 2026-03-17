import os
import json
import csv
from datetime import datetime

# Conditional import for TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

class TrainingLogger:
    """
    Unified logging engine for persistent tracking of training metrics.
    Supports CSV, JSON, and TensorBoard backends simultaneously.
    """
    def __init__(self, log_dir: str, experiment_name: str = "run", config_to_log: dict = None):
        # Generate unique experiment directory based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # File path initialization
        self.csv_path = os.path.join(self.run_dir, "history.csv")
        self.json_path = os.path.join(self.run_dir, "history.json")
        
        # Resource initialization
        self.writer = SummaryWriter(log_dir=self.run_dir) if TENSORBOARD_AVAILABLE else None
        self.history = []

        # Automatic metadata persistence
        if config_to_log:
            self.log_config(config_to_log)

    def log_config(self, config_dict: dict):
        """Serializes and saves the experiment configuration to a JSON file."""
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            # Handle non-serializable objects by converting to string
            json.dump(config_dict, f, indent=4, ensure_ascii=False, default=str)
        print(f"[LOGGER] Configuration saved to: {config_path}")

    def log_epoch(self, epoch: int, metrics: dict):
        """Logs a complete set of metrics for a specific training epoch."""
        metrics_with_epoch = {"epoch": epoch, **metrics}
        self.history.append(metrics_with_epoch)
        
        # Atomic writes to different backends
        self._write_csv(metrics_with_epoch)
        
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"Metrics/{key}", value, epoch)
                    
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4)

    def _write_csv(self, metrics: dict):
        """Internal helper for incremental CSV row appending."""
        is_new = not os.path.exists(self.csv_path)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if is_new: 
                writer.writeheader()
            writer.writerow(metrics)

    def close(self):
        """Closes all active logging streams and flushes buffers."""
        if self.writer: 
            self.writer.close()