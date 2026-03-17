import numpy as np

class EarlyStopping:
    """
    Implements early stopping to terminate training when validation loss stagnates.
    Includes robust handling for numerical instability (NaN/Inf).
    """
    def __init__(self, patience=8, delta=0.001):
        # Initialization of control parameters
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Evaluates the validation loss to determine if training should continue.
        """
        # 1. Numerical stability check
        # Treat NaN or Inf as a lack of improvement to prevent weight corruption
        if val_loss is None or np.isnan(val_loss) or np.isinf(val_loss):
            self.counter += 1
            
        # 2. First epoch initialization
        elif self.best_loss is None:
            self.best_loss = val_loss
            
        # 3. Improvement evaluation logic
        # Check if improvement exceeds the defined threshold (delta)
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter upon significant improvement
            
        # 4. Stagnation logic
        else:
            self.counter += 1 # Increment counter for sub-optimal progress

        # 5. Termination state check
        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop