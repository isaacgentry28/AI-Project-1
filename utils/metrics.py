from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score,
)


def classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, Any]:
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
    }
    if y_proba is not None:
        out['roc_auc'] = float(roc_auc_score(y_true, y_proba))
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        out['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        out['pr_curve'] = {'precision': prec.tolist(), 'recall': rec.tolist()}
    return out


def regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    return {
        'mse': float(mean_squared_error(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
    }


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
