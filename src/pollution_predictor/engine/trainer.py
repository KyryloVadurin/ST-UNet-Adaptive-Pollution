import torch
import numpy as np
from tqdm import tqdm
from .callbacks import EarlyStopping

class Trainer:
    """
    Orchestrator for the model training process.
    Manages loops, optimization, validation, and stability controls.
    """
    def __init__(self, model, optimizer, criterion, config, checkpoint_mgr, transforms):
        # Initialization of core training components
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.checkpoint_mgr = checkpoint_mgr
        self.transforms = transforms
        self.device = torch.device(config.train.device)
        self.model.to(self.device)

        # Force FP32 for absolute numerical stability
        self.use_amp = False 
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.5, patience=3
        )

    def fit(self, train_loader, val_loader, logger):
        """
        Executes the full training lifecycle across multiple epochs.
        """
        early_stopping = EarlyStopping(patience=self.config.train.early_stopping_patience)

        for epoch in range(self.config.train.epochs):
            # --- Training Phase ---
            self.model.train()
            train_loss, train_batches = 0.0, 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

            for batch in pbar:
                # Move tensors to target device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()

                # Forward pass and loss computation
                pred = self.model(batch)
                loss = self.criterion(pred, batch['target'])

                # Numerical stability check to prevent weight corruption
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                # Backpropagation and gradient management
                loss.backward()
                # Hard gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item()
                train_batches += 1

            avg_train = train_loss / max(train_batches, 1)

            # --- Validation Phase ---
            self.model.eval()
            val_loss, val_batches = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    v_loss = self.criterion(self.model(batch), batch['target'])
                    if not torch.isnan(v_loss) and not torch.isinf(v_loss):
                        val_loss += v_loss.item()
                        val_batches += 1

            avg_val = val_loss / max(val_batches, 1)

            # --- State Tracking and Checkpointing ---
            self.scheduler.step(avg_val)
            logger.log_epoch(epoch+1, {
                "train_loss": avg_train, 
                "val_loss": avg_val, 
                "lr": self.optimizer.param_groups[0]['lr']
            })
            self.checkpoint_mgr.save(
                self.model, self.optimizer, self.scheduler, 
                epoch+1, avg_val, self.config, self.transforms.get_state()
            )

            # --- Early Stopping Evaluation ---
            is_stop = early_stopping(avg_val)

            # Reporting metrics to console
            print(f"   => Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                  f"ES: {early_stopping.counter}/{early_stopping.patience}")

            if is_stop:
                print(f"\n[!] Early stopping activated at epoch {epoch+1}")
                break