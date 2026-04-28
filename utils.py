from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_runtime_device() -> str:
    return 'gpu' if tf.config.list_physical_devices('GPU') else 'cpu'


def make_model_inputs(x_cat: np.ndarray, x_num: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        'categorical_inputs': x_cat.astype(np.int32),
        'numeric_inputs': x_num.astype(np.float32),
    }


def get_predictions(
    model,
    x_cat: np.ndarray,
    x_num: np.ndarray,
    y_true: np.ndarray,
    batch_size: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = model.predict(
        make_model_inputs(x_cat, x_num),
        batch_size=batch_size,
        verbose=0,
    ).reshape(-1)
    preds = (probabilities >= threshold).astype(int)
    return y_true.astype(int), preds, probabilities


def evaluate_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': None,
        'pr_auc': None,
    }

    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
        metrics['pr_auc'] = float(average_precision_score(y_true, y_proba))

    return metrics


def report_dict(y_true, y_pred) -> Dict[str, Any]:
    return classification_report(y_true, y_pred, zero_division=0, output_dict=True)


def save_training_history(history, csv_path: Path, json_path: Path) -> pd.DataFrame:
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, 'epoch', np.arange(1, len(history_df) + 1))
    history_df.to_csv(csv_path, index=False)
    history_df.to_json(json_path, orient='records', indent=2)
    return history_df


def plot_training_curve(history_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history_df['epoch'], history_df['loss'], label='Train Loss')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
    plt.title('Curva de pérdida')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_confusion_matrix(y_true, y_pred, title: str, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f'Objeto no serializable: {type(obj)}')


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)