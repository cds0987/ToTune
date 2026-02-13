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
from ToTune.Tools.DataHf import pushdata_to_hgface,hf_login
import pandas as pd
def saved_record(point):
    df = record_to_dataframe(point)
    try:
       cloud_ = load_dataset(point['cloud_ds'],split = 'train').to_pandas()
       hf_login(point)
       df = pd.concat([cloud_,df],ignore_index=True)
       pushdata_to_hgface(point['cloud_ds'],cloud_ds)
    except:
       print('Cannot upload record onto HuggingFace saved on local')
       save_local_path = point.get('local_save_path',None)
       if save_local_path is not None:
          df.to_csv(save_local_path,index = False)
          print(f"Saved record locally at {save_local_path}")  
       else:
          df.to_csv('training_record.csv',index = False)
          print(f"Saved record locally at training_record.csv")
       