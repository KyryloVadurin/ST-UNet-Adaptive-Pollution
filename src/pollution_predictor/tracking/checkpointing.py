import torch
import os
import heapq

class CheckpointManager:
    """
    Manages the saving and rotation of model checkpoints.
    Keeps the top-K best performing models based on validation metrics.
    """
    def __init__(self, save_dir: str, top_k: int = 3):
        # Initialize storage directory and ranking heap
        self.save_dir = save_dir
        self.top_k = top_k
        self.best_metrics = [] # List of tuples: (metric, filepath)
        os.makedirs(self.save_dir, exist_ok=True)

    def save(self, model, optimizer, scheduler, epoch, val_loss, config, transforms_state, is_best=False, filename=None):
        """
        Saves current training state and updates the top-K ranking.
        """
        # Assemble state dictionary with metadata and weights
        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'val_loss': val_loss,
            'config': config.model_dump(),
            'transforms_state': transforms_state
        }

        # Always maintain the absolute latest model for resuming
        last_path = os.path.join(self.save_dir, "last_model.pth")
        torch.save(state, last_path)

        # Top-K rotation logic using a min-heap
        if len(self.best_metrics) < self.top_k:
            # Add to heap if capacity not reached
            best_path = os.path.join(self.save_dir, f"best_model_epoch_{epoch}_loss_{val_loss:.4f}.pth")
            torch.save(state, best_path)
            heapq.heappush(self.best_metrics, (-val_loss, best_path)) 
        elif -val_loss > self.best_metrics[0][0]:
            # Replace worst in Top-K if current model is better
            popped_loss, popped_path = heapq.heappop(self.best_metrics)
            if os.path.exists(popped_path):
                os.remove(popped_path) 

            best_path = os.path.join(self.save_dir, f"best_model_epoch_{epoch}_loss_{val_loss:.4f}.pth")
            torch.save(state, best_path)
            heapq.heappush(self.best_metrics, (-val_loss, best_path))

    def save_emergency(self, model, optimizer, epoch, config, transforms_state):
        """
        Minimum viable save for immediate crash recovery.
        """
        try:
            state = {
                'epoch': epoch, 
                'model_state': model.state_dict(), 
                'optimizer_state': optimizer.state_dict()
            }
            torch.save(state, os.path.join(self.save_dir, "crash_recovery_model.pth"))
            print("[EMERGENCY SAVE] Successfully saved.")
        except Exception as e:
            print(f"[EMERGENCY SAVE] ERROR: {e}")

    @staticmethod
    def load(filepath: str, device='cpu'):
        """
        Static method to deserialize model artifacts.
        """
        return torch.load(filepath, map_location=device, weights_only=False)