import numpy as np
from sklearn.metrics import (
    f1_score, mean_squared_error, r2_score, mean_absolute_error,
    precision_score, recall_score, max_error
)
from scipy.stats import pearsonr, wasserstein_distance
import warnings

# Safe import for SSIM metric
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image is not installed. SSIM metric will be skipped. Install via: pip install scikit-image")

def get_center_of_mass(img: np.ndarray) -> tuple:
    """Calculates the center of mass (centroid) of pollution intensity."""
    total = img.sum() + 1e-8
    X, Y = np.indices(img.shape)
    x_c = (X * img).sum() / total
    y_c = (Y * img).sum() / total
    return x_c, y_c

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Comprehensive evaluation of the prediction across 4 dimensions:
    Regression, Structure, Localization, and Classification (Segmentation).
    """
    metrics = {}
    
    # Data preparation and binarization
    y_t_flat = y_true.flatten()
    y_p_flat = y_pred.flatten()
    
    y_true_bin = (y_true > threshold).flatten()
    y_pred_bin = (y_pred > threshold).flatten()

    # 1. REGRESSION METRICS (Pixel-wise)
    metrics["RMSE"] = float(np.sqrt(mean_squared_error(y_t_flat, y_p_flat)))
    metrics["MAE"] = float(mean_absolute_error(y_t_flat, y_p_flat))
    metrics["Max_Error"] = float(max_error(y_t_flat, y_p_flat))
    
    # Symmetric Mean Absolute Percentage Error (sMAPE)
    metrics["sMAPE"] = float(np.mean(2.0 * np.abs(y_p_flat - y_t_flat) / (np.abs(y_t_flat) + np.abs(y_p_flat) + 1e-8)) * 100.0)
    
    metrics["R2"] = float(r2_score(y_t_flat, y_p_flat))
    
    # Pearson Correlation Coefficient
    if np.std(y_p_flat) < 1e-9 or np.std(y_t_flat) < 1e-9:
        metrics["Pearson_r"] = 0.0
    else:
        metrics["Pearson_r"] = float(pearsonr(y_t_flat, y_p_flat)[0])

    # 2. STRUCTURAL AND SPATIAL METRICS
    # Peak Signal-to-Noise Ratio (PSNR)
    data_range = float(y_true.max() - y_true.min()) if y_true.max() > y_true.min() else 1.0
    mse = metrics["RMSE"]**2
    metrics["PSNR"] = float(10 * np.log10((data_range**2) / (mse + 1e-8)))

    # Structural Similarity Index (SSIM)
    if SKIMAGE_AVAILABLE:
        metrics["SSIM"] = float(ssim(y_true, y_pred, data_range=data_range))
    else:
        metrics["SSIM"] = 0.0

    # Cosine Similarity
    dot_product = np.dot(y_t_flat, y_p_flat)
    norm_t, norm_p = np.linalg.norm(y_t_flat), np.linalg.norm(y_p_flat)
    metrics["Cosine_Sim"] = float(dot_product / (norm_t * norm_p + 1e-8))

    # Wasserstein Distance (Earth Mover's Distance)
    metrics["Wasserstein_Dist"] = float(wasserstein_distance(y_t_flat, y_p_flat))

    # 3. LOCALIZATION METRICS (Epicenters)
    # Localization Error of the Maximum intensity pixel (LE_Max)
    t_max = np.unravel_index(np.argmax(y_true), y_true.shape)
    p_max = np.unravel_index(np.argmax(y_pred), y_pred.shape)
    metrics["LE_Max_px"] = float(np.sqrt((t_max[0]-p_max[0])**2 + (t_max[1]-p_max[1])**2))

    # Center of Mass Error (CME)
    cx_t, cy_t = get_center_of_mass(y_true)
    cx_p, cy_p = get_center_of_mass(y_pred)
    metrics["CME_px"] = float(np.sqrt((cx_t - cx_p)**2 + (cy_t - cy_p)**2))

    # 4. CLASSIFICATION METRICS (Hazard Zone Segmentation)
    intersection = np.logical_and(y_true_bin, y_pred_bin).sum()
    union = np.logical_or(y_true_bin, y_pred_bin).sum()
    
    metrics["IoU"] = float(intersection / (union + 1e-8))
    metrics["F1"] = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
    metrics["Precision"] = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
    metrics["Recall"] = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
    
    # Specificity (True Negative Rate)
    tn = np.logical_and(~y_true_bin, ~y_pred_bin).sum()
    fp = np.logical_and(~y_true_bin, y_pred_bin).sum()
    metrics["Specificity"] = float(tn / (tn + fp + 1e-8))

    return metrics