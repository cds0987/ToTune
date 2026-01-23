import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)

def evaluate_classification(labels, predictions):
    """
    labels: list or array of ground-truth labels
    predictions: list or array of predicted labels
    """

    y_true = np.asarray(labels)
    y_pred = np.asarray(predictions)

    # -----------------------------
    # 1. Core metrics
    # -----------------------------
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision (Macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (Macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-score (Macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision (Weighted)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall (Weighted)": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-score (Weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Matthews Corrcoef (MCC)": matthews_corrcoef(y_true, y_pred),
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
    }

    metrics_df = (
        pd.DataFrame.from_dict(metrics, orient="index", columns=["Score"])
          .reset_index()
          .rename(columns={"index": "Metric"})
    )

    # -----------------------------
    # 2. Confusion matrix
    # -----------------------------
    cm = confusion_matrix(y_true, y_pred)

    # Support (true samples per class)
    support = cm.sum(axis=1)

    # Normalized confusion matrix (row-wise)
    cm_norm = cm / support[:, None]

    labels_sorted = np.unique(y_true)

    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{i}" for i in labels_sorted],
        columns=[f"Pred_{i}" for i in labels_sorted],
    )

    cm_norm_df = pd.DataFrame(
        cm_norm,
        index=[f"True_{i}" for i in labels_sorted],
        columns=[f"Pred_{i}" for i in labels_sorted],
    )

    cm_norm_df["Support"] = support

    # -----------------------------
    # 3. Classification report
    # -----------------------------
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T

    return {
        "metrics": metrics_df.round(4),
        "confusion_matrix": cm_df,
        "confusion_matrix_normalized": cm_norm_df.round(4),
        "classification_report": report_df.round(4),
    }


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm_df, title="Confusion Matrix",
                          cell_size=1.2, max_size=20):
    data = cm_df.copy()

    # Remove Support column if present
    if "Support" in data.columns:
        data = data.drop(columns=["Support"])

    n_classes = data.shape[0]

    # Auto figure size
    fig_width = min(max_size, max(6, n_classes * cell_size))
    fig_height = min(max_size, max(5, n_classes * (cell_size*0.5)))

    is_float = np.issubdtype(data.values.dtype, np.floating)
    fmt = ".2f" if is_float else "d"

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=True,
        linewidths=0.5,
        square=True
    )

    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()
