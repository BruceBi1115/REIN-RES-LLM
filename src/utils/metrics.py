import numpy as np

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def smape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(100.0 * np.mean(2*np.abs(y_pred - y_true) / (np.abs(y_true)+np.abs(y_pred)+eps)))

def skill_score(final_mse, base_mse, eps=1e-12):
    final_mse = float(final_mse)
    base_mse = float(base_mse)
    if base_mse <= eps:
        return 0.0
    return float(1.0 - final_mse / base_mse)
