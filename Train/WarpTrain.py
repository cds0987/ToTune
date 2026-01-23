from ToTune.Peft.Utils import load_peft_model
from ToTune.models.Utils import load_sequence_classification_model
from ToTune.models.SequenceClassification import SequenceClassification

def print_point(point,title = 'TRAINING CONFIG'):
    print(f"\n===== {title} =====")
    for k, v in point.items():
        print(f"{k:<22}: {v}")
    print("===========================\n")

def sequence_classification_model(point):
    load_in_4bit = point['load_in_4bit']
    num_labels = point['num_labels']
    use_gradient_checkpointing = point['use_gradient_checkpointing']
    model_name = point['model_name']
    max_seq = point['max_seq']
    Used_model, tokenizer = load_sequence_classification_model(
        model_name, num_labels, load_in_4bit=load_in_4bit,use_gradient_checkpointing = use_gradient_checkpointing
    )
    peft_config = point['peft_config'] if 'peft_config' in point else None
    if peft_config is None:
        peft_config = {}
        peft_config['target_modules'] = ['Not used']
        peft_config['r'] = -1
    else:
        adaptation = peft_config['adaptation']
        peft_config.pop("adaptation", None)
        Used_model,adaptation = load_peft_model(Used_model,adaptation = adaptation, **peft_config)
        peft_config = adaptation
    SeqCls = SequenceClassification(model_name,Used_model,tokenizer,peft_config,max_seq)
    return SeqCls

def prepare_data(SeqCls,point):
  train_ds, test_ds, text_col, labels_col = point['train_ds'], point['test_ds'], point['text_col'], point['labels_col']
  SeqCls.preprocess(train_ds,test_ds,text_col,labels_col)

def trainersetting(SeqCls,point):
  args = point['trainargs'] if 'trainargs' in point else None
  SeqCls.prepare_trainer(args)

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

from ToTune.Tools.EvaluatePerformance import evaluate_classification,plot_confusion_matrix,evaluate_probabilities
from IPython.display import display
def postprocess(SeqCls):
    SeqCls.train_test()

    print_tunning(SeqCls.output)

    labels = SeqCls.output["labels"]
    preds = SeqCls.output["preds"]
    probs = SeqCls.output["probs"]

    results = evaluate_classification(labels=labels, predictions=preds)
    results["prob_metrics"] = evaluate_probabilities(labels=labels, probs=probs)

    SeqCls.output["evaluation"] = results

    print_evaluation_report(results)


def train_SeqCls(point):
  print_point(point)
  SeqCls = sequence_classification_model(point)
  prepare_data(SeqCls,point)
  trainersetting(SeqCls,point)
  postprocess(SeqCls)
  return SeqCls