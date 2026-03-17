import numpy as np
import json
import os
from .metrics import evaluate_predictions
from .baselines import ClassicalBaseline

class SystemEvaluator:
    """
    Core evaluation engine that compares the Neural Network model against 
    a classical baseline using various spatio-temporal metrics.
    """
    def __init__(self, predictor, test_loader, output_dir):
        # Initialization of predictor, data source, and baseline components
        self.predictor = predictor
        self.test_loader = test_loader
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.baseline = ClassicalBaseline()

    def run_evaluation(self, use_stochastic=False, n_samples=10):
        """
        Executes a full evaluation pass on the provided test dataset.
        Supports both deterministic and stochastic (Monte Carlo) inference modes.
        """
        print(f"\n=== FINAL EVALUATION (mode: {'stochastic' if use_stochastic else 'deterministic'}) ===")
        nn_metrics, base_metrics = [], []

        # Feature usage check
        model_uses_wind = getattr(self.predictor.model, 'use_wind', True)

        # Batch processing loop
        for batch in self.test_loader:
            norm_readings = batch['readings'].numpy()
            coords = batch['coords'].numpy()
            target = batch['target'].squeeze(1).numpy()

            # Denormalization logic
            t = self.predictor.transforms
            raw_readings = np.expm1(norm_readings * t.r_std + t.r_mean)

            # Wind data preparation
            raw_wind = None
            if model_uses_wind and 'wind' in batch:
                norm_wind = batch['wind'].numpy()
                raw_wind = norm_wind * t.max_wind_speed

            # Predictor inference execution
            if use_stochastic:
                pred_nn, _ = self.predictor.predict_stochastic(
                    raw_readings.transpose(0, 2, 1), coords, raw_wind, n_samples=n_samples
                )
            else:
                pred_nn = self.predictor.predict_deterministic(
                    raw_readings.transpose(0, 2, 1), coords, raw_wind
                )

            # Per-sample metric calculation loop
            for i in range(raw_readings.shape[0]):
                nn_metrics.append(evaluate_predictions(target[i], pred_nn[i]))
                p_base = self.baseline.predict(
                    raw_readings[i].T, coords[i], 
                    raw_wind[i] if raw_wind is not None else None, target[i].shape
                )
                base_metrics.append(evaluate_predictions(target[i], p_base))

        # Result verification
        if not nn_metrics:
            print("Error: Test dataset is empty.")
            return

        # Statistical aggregation (averaging)
        avg_nn = {k: np.mean([m[k] for m in nn_metrics]) for k in nn_metrics[0]}
        avg_base = {k: np.mean([m[k] for m in base_metrics]) for k in base_metrics[0]}

        # Comprehensive report serialization
        report = {"Neural_Network": avg_nn, "Classical_Baseline": avg_base}
        mode_str = 'stochastic' if use_stochastic else 'deterministic'
        report_path = os.path.join(self.output_dir, f"test_report_{mode_str}.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        # Console reporting
        self._print_summary("Classical Baseline", avg_base)
        self._print_summary("Neural Network Model (NN)", avg_nn)
        print(f"\n[i] Full extended report saved to: {report_path}")

    def _print_summary(self, title: str, metrics: dict):
        """
        Groups and prints evaluation metrics in a structured format.
        """
        print(f"\n--- {title} ---")
        
        # Regression and Structural group
        print("  Regression and Structure:")
        print(f"    RMSE:       {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f}")
        print(f"    R2 Score:   {metrics['R2']:.4f} | Pearson_r: {metrics['Pearson_r']:.4f}")
        print(f"    SSIM:       {metrics['SSIM']:.4f} | PSNR: {metrics['PSNR']:.2f} dB")
        print(f"    Wasserstein:{metrics['Wasserstein_Dist']:.4f}")
        
        # Epicenter Localization group
        print("  Epicenter Localization:")
        print(f"    LE_Max_px:  {metrics['LE_Max_px']:.1f} px (Absolute maximum error)")
        print(f"    CME_px:     {metrics['CME_px']:.1f} px (Cloud center of mass error)")
        
        # Hazard Zone Segmentation group (Thresholded at 0.5)
        print("  Zone Segmentation (Threshold=0.5):")
        print(f"    F1 Score:   {metrics['F1']:.4f} | IoU: {metrics['IoU']:.4f}")
        print(f"    Precision:  {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f}")