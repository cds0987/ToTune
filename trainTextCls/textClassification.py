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




