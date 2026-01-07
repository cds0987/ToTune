import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score,
    classification_report,
    top_k_accuracy_score
)

def classification_evaluation(
    y_true,
    y_pred,
    y_proba=None,
    class_names=None,
    multilabel=False,
    top_k=(1, 3, 5),
    show_confusion_matrix=True,
    export_csv=None,
    export_latex=None,
    cmap="Blues"
):
    """
    Comprehensive evaluation for multi-class & multi-label classification.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels (shape: [n_samples] or [n_samples, n_classes])
    y_pred : array-like
        Predicted labels (same shape as y_true)
    y_proba : array-like, optional
        Predicted probabilities (required for top-k accuracy)
    class_names : list, optional
        Class names
    multilabel : bool, default=False
        Whether task is multi-label (one-vs-rest)
    top_k : tuple, default=(1,3,5)
        Top-k accuracies to compute (multi-class only)
    show_confusion_matrix : bool
        Whether to display confusion matrix
    export_csv : str, optional
        Path to save CSV summary
    export_latex : str, optional
        Path to save LaTeX table
    cmap : str
        Colormap for confusion matrix

    Returns
    -------
    per_class_df : pd.DataFrame
    summary_df : pd.DataFrame
    sklearn_style_df : pd.DataFrame
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # -----------------------------
    # Class handling
    # -----------------------------
    if not multilabel:
        labels = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(labels)
    else:
        n_classes = y_true.shape[1]
        labels = np.arange(n_classes)

    if class_names is None:
        class_names = [f"Class {i}" for i in labels]

    # -----------------------------
    # Per-class metrics
    # -----------------------------
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )

    per_class_df = pd.DataFrame({
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Support": support
    })

    per_class_df[["Precision", "Recall", "F1-score"]] = \
        per_class_df[["Precision", "Recall", "F1-score"]].round(4)

    # -----------------------------
    # Global metrics
    # -----------------------------
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred) if not multilabel else np.nan,
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred) if not multilabel else np.nan,
        "MCC": matthews_corrcoef(y_true.flatten(), y_pred.flatten())
    }

    for avg in ["macro", "micro", "weighted"]:
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        metrics[f"Precision ({avg})"] = p
        metrics[f"Recall ({avg})"] = r
        metrics[f"F1-score ({avg})"] = f

    # -----------------------------
    # Top-K accuracy (multi-class)
    # -----------------------------
    if (y_proba is not None) and (not multilabel):
        for k in top_k:
            if k <= y_proba.shape[1]:
                metrics[f"Top-{k} Accuracy"] = top_k_accuracy_score(
                    y_true, y_proba, k=k
                )

    summary_df = pd.DataFrame(
        {"Metric": metrics.keys(), "Value": metrics.values()}
    )
    summary_df["Value"] = summary_df["Value"].round(4)

    # -----------------------------
    # Sklearn-style report table
    # -----------------------------
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    sklearn_style_df = (
        pd.DataFrame(report_dict)
        .transpose()
        .round(4)
    )

    # -----------------------------
    # Confusion matrix (disabled for multilabel)
    # -----------------------------
    if show_confusion_matrix and not multilabel:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(1.2 * n_classes, 1 * n_classes))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # Export
    # -----------------------------
    if export_csv:
        summary_df.to_csv(export_csv, index=False)

    if export_latex:
        summary_df.to_latex(export_latex, index=False)

    return per_class_df, summary_df, sklearn_style_df