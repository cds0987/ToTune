from ToTune.models.load import Load_sequence_classification_model,loadSFTgemma3,loadSFTQwen3Alpaca,loadEmbeddingSeqCls
def load_sequence_classification_model(point):
    procedure = point['procedure'] if 'procedure' in point else 'SeqCls'
    if procedure == 'SeqCls':
        SeqCls = Load_sequence_classification_model(point)
    elif procedure == 'SFTGemma3':
        SeqCls = loadSFTgemma3(point)
    elif procedure == 'SFTQwen3Alpaca':
        SeqCls = loadSFTQwen3Alpaca(point)
    elif procedure == 'EmbeddingSeqCls':
        SeqCls = loadEmbeddingSeqCls(point)
    return SeqCls


def prepare_data(SeqCls,point):
  train_ds, test_ds, texts_col, labels_col = point['train_ds'], point['test_ds'], point['texts_col'], point['labels_col']
  SeqCls.preprocess(train_ds,test_ds,texts_col,labels_col)

def trainersetting(SeqCls,point):
  args = point['trainargs'] if 'trainargs' in point else None
  SeqCls.prepare_trainer(args)

from ToTune.trainTextCls.utils import evaluate_classification,plot_confusion_matrix,evaluate_probabilities
from ToTune.trainTextCls.utils import print_tunning,print_point,print_evaluation_report
from ToTune.trainTextCls.utils import encode_labels_and_preds
from IPython.display import display
def SeqCls_postprocess(SeqCls):
    labels = SeqCls.output["labels"]
    preds = SeqCls.output["preds"]
    probs = SeqCls.output["probs"] if "probs" in SeqCls.output else None

    
    if probs is not None:
        results = evaluate_classification(labels=labels, predictions=preds)
        results["prob_metrics"] = evaluate_probabilities(labels=labels, probs=probs)
        
    else:
        
        y_true, y_pred, label2id, id2label = encode_labels_and_preds(labels, preds)
        results = evaluate_classification(labels=y_true, predictions=y_pred)
        SeqCls.output["label2id"] = label2id
        SeqCls.output["id2label"] = id2label
        SeqCls.output["y_true"] = y_true
        SeqCls.output["y_pred"] = y_pred
    SeqCls.output["evaluation"] = results
    print_evaluation_report(results)
    
    
from ToTune.tools.record import get_current_datetime, dataset_agreegate
from ToTune.trainTextCls.utils import flatten_evaluation_dict,saved_record,savedModel
def train_SeqCls(point):
    point['tunemode'] = 'TextClassification'
    print_point(point)
    print(f"\n===== Load Model =====")
    SeqCls = load_sequence_classification_model(point)
    print(f"\n===== Prepare Data =====")
    prepare_data(SeqCls,point)
    print(f"\n===== Train Model =====")
    trainersetting(SeqCls,point)
    SeqCls.train_test()
    SeqCls_postprocess(SeqCls)
    point['timestamp'] = get_current_datetime()
    dataset_agreegate(point)
    point['train_report'] = SeqCls.output['train_report']
    point['evaluation'] = flatten_evaluation_dict(SeqCls.output['evaluation'])
    saved_record(point)
    point['model'] = SeqCls.model
    point['tokenizer'] = SeqCls.tokenizer
    savedModel(point)
    return SeqCls




import numpy as np

def print_classification_results(
    texts,
    preds,
    probs=None,
    label_names=None,
    top_k=3,
    max_text_len=120
):
    """
    Flexible print-only classification result viewer.

    Args:
        texts (list[str]): input texts
        preds (list[int|str]): predicted class indices or names
        probs (list[list|dict], optional): probability distributions
        label_names (list[str], optional): class names
        top_k (int): number of top classes to show
        max_text_len (int): truncate long texts
    """

    for i, text in enumerate(texts):
        pred = preds[i]
        prob = probs[i] if probs is not None else None

        print("\n" + "=" * 90)
        print(f"Sample {i}")
        print("-" * 90)

        # Text
        short_text = text[:max_text_len] + ("..." if len(text) > max_text_len else "")
        print(f"Text: {short_text}\n")

        # ---- Prediction handling (int or str) ----
        if isinstance(pred, int):
            pred_name = (
                label_names[pred]
                if label_names is not None and pred < len(label_names)
                else f"Class_{pred}"
            )
        else:
            pred_name = str(pred)

        print(f"Predicted Class : {pred_name}")

        # ---- Probability handling ----
        if prob is None:
            print("Confidence      : N/A (no probabilities provided)")
            continue

        # Dict-style probabilities
        if isinstance(prob, dict):
            sorted_items = sorted(prob.items(), key=lambda x: x[1], reverse=True)

            confidence = prob.get(pred_name, max(prob.values()))
            print(f"Confidence      : {confidence:.6f}\n")

            print(f"Top-{top_k} Classes:")
            for cls, p in sorted_items[:top_k]:
                marker = "<==" if cls == pred_name else ""
                print(f"  {cls:<15} : {p:.6f} {marker}")

        # List / ndarray probabilities
        else:
            prob = np.asarray(prob)
            topk_idx = prob.argsort()[-top_k:][::-1]

            confidence = prob[pred] if isinstance(pred, int) and pred < len(prob) else prob.max()
            print(f"Confidence      : {confidence:.6f}\n")

            print(f"Top-{top_k} Classes:")
            for idx in topk_idx:
                cls_name = (
                    label_names[idx]
                    if label_names is not None and idx < len(label_names)
                    else f"Class_{idx}"
                )
                marker = "<==" if isinstance(pred, int) and idx == pred else ""
                print(f"  {cls_name:<15} : {prob[idx]:.6f} {marker}")

    print("\n" + "=" * 90)
    print("End of predictions")