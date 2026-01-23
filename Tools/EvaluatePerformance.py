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



from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    top_k_accuracy_score,
)

def evaluate_probabilities(labels, probs):
    """
    labels: array-like of shape (n_samples,)
    probs : array-like of shape (n_samples, n_classes)
    """

    y_true = np.asarray(labels)
    probs = np.asarray(probs)

    n_classes = probs.shape[1]
    is_binary = n_classes == 2

    metrics = {}

    # -----------------------------
    # 1. Calibration / confidence
    # -----------------------------
    metrics["Log Loss"] = log_loss(y_true, probs)

    if is_binary:
        metrics["Brier Score"] = brier_score_loss(y_true, probs[:, 1])

    # -----------------------------
    # 2. Ranking quality
    # -----------------------------
    if is_binary:
        metrics["ROC-AUC"] = roc_auc_score(y_true, probs[:, 1])
        metrics["PR-AUC"] = average_precision_score(y_true, probs[:, 1])
    else:
        metrics["ROC-AUC (OvR Macro)"] = roc_auc_score(
            y_true, probs, multi_class="ovr", average="macro"
        )
        metrics["PR-AUC (Macro)"] = average_precision_score(
            y_true, probs, average="macro"
        )

    # -----------------------------
    # 3. Top-k accuracy (prob-based)
    # -----------------------------
    for k in [1, 3, 5]:
        if k <= n_classes:
            metrics[f"Top-{k} Accuracy"] = top_k_accuracy_score(
                y_true, probs, k=k
            )

    # -----------------------------
    # 4. Confidence statistics
    # -----------------------------
    max_conf = probs.max(axis=1)

    metrics["Mean Confidence"] = max_conf.mean()
    metrics["Median Confidence"] = np.median(max_conf)
    metrics["Confidence Std"] = max_conf.std()

    # -----------------------------
    # 5. Uncertainty (entropy)
    # -----------------------------
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    metrics["Mean Entropy"] = entropy.mean()
    metrics["Entropy Std"] = entropy.std()

    return (
        pd.DataFrame.from_dict(metrics, orient="index", columns=["Score"])
        .reset_index()
        .rename(columns={"index": "Metric"})
        .round(4)
    )