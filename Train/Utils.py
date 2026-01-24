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