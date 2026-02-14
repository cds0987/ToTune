from ToTune.tools.evaluatePerformance import evaluate_classification,plot_confusion_matrix,evaluate_probabilities
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


from datasets import Dataset, DatasetDict
from typing import Union, Dict, List


def print_dataset_demo(
    data: Union[Dataset, DatasetDict, Dict[str, Dataset], List[Dataset]],
    splits: List[str] = None,
    n_samples: int = 3,
    columns: List[str] = None,
    max_text_len: int = 200,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Universal demo printer for HuggingFace datasets.

    Supports:
      - Dataset
      - DatasetDict
      - dict[str, Dataset]
      - list[Dataset]

    Args:
        data: Dataset / DatasetDict / dict / list
        splits: Split names to show (only for DatasetDict or dict)
        n_samples: Number of samples per split
        columns: Columns to display (None = auto-detect)
        max_text_len: Max characters per text field
        shuffle: Whether to shuffle before sampling
        seed: Random seed
    """
    print(f"\n===== Dataset Preview =====")
    def _shorten(val):
        if isinstance(val, str) and len(val) > max_text_len:
            return val[:max_text_len] + "..."
        return val

    def _print_dataset(ds: Dataset, name: str):
        print(f"\n{'=' * 70}")
        print(f"SPLIT: {name} | rows={len(ds)}")
        print(f"{'=' * 70}")

        sample_ds = ds
        if shuffle:
            sample_ds = sample_ds.shuffle(seed=seed)

        sample_ds = sample_ds.select(range(min(n_samples, len(sample_ds))))

        cols = columns or sample_ds.column_names

        for i, row in enumerate(sample_ds):
            print(f"\n[{name} sample {i + 1}]")
            for col in cols:
                if col not in row:
                    continue
                print(f"Column: {col}: {_shorten(row[col])}")

    # ---- Dispatcher ----
    if isinstance(data, Dataset):
        _print_dataset(data, "dataset")

    elif isinstance(data, (DatasetDict, dict)):
        split_names = splits or list(data.keys())
        for split in split_names:
            if split in data:
                _print_dataset(data[split], split)

    elif isinstance(data, list):
        for idx, ds in enumerate(data):
            _print_dataset(ds, f"dataset_{idx}")

    else:
        raise TypeError(
            "Unsupported input type. Expected Dataset, DatasetDict, dict, or list."
        )
        
import numpy as np

def flatten_evaluation_dict(evaluation_dict):
    """
    Flatten SeqCls.output['evaluation'] into a single flat dictionary.
    Converts numpy types to pure Python scalars.
    """

    flat = {}

    # -----------------------------------
    # 1. Main metrics dataframe
    # -----------------------------------
    if "metrics" in evaluation_dict:
        df = evaluation_dict["metrics"]
        for _, row in df.iterrows():
            key = f"metric__{row['Metric']}"
            flat[key] = row["Score"]

    # -----------------------------------
    # 2. Probability metrics dataframe
    # -----------------------------------
    if "prob_metrics" in evaluation_dict:
        df = evaluation_dict["prob_metrics"]
        for _, row in df.iterrows():
            key = f"prob__{row['Metric']}"
            flat[key] = row["Score"]

    # -----------------------------------
    # 3. Confusion matrix
    # -----------------------------------
    if "confusion_matrix" in evaluation_dict:
        cm = evaluation_dict["confusion_matrix"]
        for row_idx in cm.index:
            for col in cm.columns:
                key = f"cm__{row_idx}__{col}"
                flat[key] = cm.loc[row_idx, col]

    # -----------------------------------
    # 4. Normalized confusion matrix
    # -----------------------------------
    if "confusion_matrix_normalized" in evaluation_dict:
        cmn = evaluation_dict["confusion_matrix_normalized"]
        for row_idx in cmn.index:
            for col in cmn.columns:
                key = f"cm_norm__{row_idx}__{col}"
                flat[key] = cmn.loc[row_idx, col]

    # -----------------------------------
    # 5. Classification report
    # -----------------------------------
    if "classification_report" in evaluation_dict:
        cr = evaluation_dict["classification_report"]
        for row_idx in cr.index:
            for col in cr.columns:
                key = f"report__{row_idx}__{col}"
                flat[key] = cr.loc[row_idx, col]

    # -----------------------------------
    # Convert numpy scalars → Python scalars
    # -----------------------------------
    flat = {
        k: (v.item() if isinstance(v, np.generic) else v)
        for k, v in flat.items()
    }

    return flat


from datasets import Dataset, concatenate_datasets,load_dataset
from ToTune.tools.dataHf import pushdata_to_hgface,hf_login
from ToTune.tools.record import record_to_dataframe
import pandas as pd
def saved_record(point):
    print("\n===== Saved Record Training =====")
    df = record_to_dataframe(point)
    try:
       cloud_ = load_dataset(point['cloud_ds'],split = 'train').to_pandas()
       hf_login(point)
       df = pd.concat([cloud_,df],ignore_index=True)
       pushdata_to_hgface(point['cloud_ds'],df,token = point['token'])
    except:
       print('Cannot upload record onto HuggingFace saved on local')
       save_local_path = point.get('local_save_path',None)
       if save_local_path is not None:
          df.to_csv(save_local_path,index = False)
          print(f"Saved record locally at {save_local_path}")  
       else:
          df.to_csv('training_record.csv',index = False)
          print(f"Saved record locally at training_record.csv")

from ToTune.tools.modelhf import save_ModelHgface
def savedModel(point):
    if point['model_saved'] is None:
         print("\n No model name provided, skipping model saving.")
         return
    print("\n===== Saved Model =====")
    try:
        token = point['token']
        username = point['username']
        model_name = point['model_saved']
        save_ModelHgface(point['model'], point['tokenizer'], username, model_name, token)
    except Exception as e:
        print(f"Failed to save model to HuggingFace: {e}")
        print("Model  saved locally instead.")
        point['model'].save_pretrained(f"{point['model_saved']}_local")
        point['tokenizer'].save_pretrained(f"{point['model_saved']}_local")    
        
        
        
        
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