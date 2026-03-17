from pydantic import BaseModel, Field
from typing import Tuple, Optional, Dict, Any

class DataConfig(BaseModel):
    """
    Configuration for data loading, splitting, and preprocessing.
    """
    dataset_dir: str
    batch_size: int = 16
    grid_size: Tuple[int, int] = (50, 50)
    num_workers: int = 0
    val_split: float = 0.15
    test_split: float = 0.05
    random_seed: int = 42
    max_wind_speed: Optional[float] = None 
    use_augmentations: bool = True
    use_wind: bool = True

class ModelConfig(BaseModel):
    """
    Configuration for neural network architecture and hyperparameters.
    """
    architecture: str = "st_unet"
    time_steps: int = 48
    use_wind: bool = True
    # Parameter dictionary for specific architectural settings (e.g., hidden layers, dropout)
    params: Dict[str, Any] = Field(default_factory=lambda: {"hidden_dim": 64, "dropout_rate": 0.2})

class TrainConfig(BaseModel):
    """
    Configuration for the training loop, optimization, and loss functions.
    """
    epochs: int = 30
    learning_rate: float = 2e-4
    early_stopping_patience: int = 10
    mse_weight: float = 1.0
    dice_weight: float = 5.0
    focus_factor: float = 15.0
    device: str = "cuda"
    accumulation_steps: int = 1
    resume_from: Optional[str] = None
    loss_warmup_epochs: int = 5
    # Switches for Fine-Tuning and Transfer Learning
    freeze_encoder: bool = False
    transfer_learning: bool = False # If True, only model weights are loaded without optimizer state

class TrackerConfig(BaseModel):
    """
    Configuration for logging and checkpoint management.
    """
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "pollution_run"

class AppConfig(BaseModel):
    """
    Root configuration object aggregating all sub-configs.
    """
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    tracker: TrackerConfig = TrackerConfig()