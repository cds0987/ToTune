from ToTune.Tools.EvaluatePerformance import evaluate_classification,plot_confusion_matrix,evaluate_probabilities
from IPython.display import display

def print_point(point,title = 'TRAINING CONFIG'):
    print(f"\n===== {title} =====")
    for k, v in point.items():
        print(f"{k:<22}: {v}")
    print("===========================\n")
def print_evaluation_report(results):
    print("\n" + "=" * 60)
    print("📊 CLASSIFICATION PERFORMANCE SUMMARY")
    print("=" * 60)

    print("\n▶ Overall Metrics")
    display(results["metrics"])

    print("\n▶ Normalized Confusion Matrix")
    plot_confusion_matrix(results["confusion_matrix_normalized"])

    print("\n▶ Per-Class Classification Report")
    display(results["classification_report"])

    if "prob_metrics" in results:
        print("\n" + "=" * 60)
        print("📈 PROBABILITY-BASED EVALUATION")
        print("=" * 60)
        display(results["prob_metrics"])

    print("\n" + "=" * 60)

def print_tunning(output):
    print_point(output["Tuner_arg"], "Tuner Config")
    print_point(output["adaptation"], "Adaptation Config")


def encode_labels_and_preds(labels, preds, outlier_name="__OUTLIER__"):
    """
    labels: list[str] - ground truth labels
    preds : list[str] - predicted labels

    Returns:
        y_true : list[int]
        y_pred : list[int]
        label2id : dict[str, int]
        id2label : dict[int, str]
    """

    # 1. Build label mapping from ground truth only
    unique_labels = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    outlier_id = len(label2id)
    id2label[outlier_id] = outlier_name

    # 2. Encode ground truth
    y_true = [label2id[label] for label in labels]

    # 3. Encode predictions with outlier handling
    y_pred = [
        label2id[p] if p in label2id else outlier_id
        for p in preds
    ]

    return y_true, y_pred, label2id, id2label