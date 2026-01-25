from ToTune.models.Load import Load_sequence_classification_model,loadSFTgemma3,loadSFTQwen3Alpaca
def load_sequence_classification_model(point):
    procedure = point['procedure'] if 'procedure' in point else 'SeqCls'
    if procedure == 'SeqCls':
        SeqCls = Load_sequence_classification_model(point)
    elif procedure == 'SFTGemma3':
        SeqCls = loadSFTgemma3(point)
    elif procedure == 'SFTQwen3Alpaca':
        SeqCls = loadSFTQwen3Alpaca(point)
    return SeqCls


def prepare_data(SeqCls,point):
  train_ds, test_ds, text_col, labels_col = point['train_ds'], point['test_ds'], point['text_col'], point['labels_col']
  SeqCls.preprocess(train_ds,test_ds,text_col,labels_col)

def trainersetting(SeqCls,point):
  args = point['trainargs'] if 'trainargs' in point else None
  SeqCls.prepare_trainer(args)

from ToTune.Train.Utils import evaluate_classification,plot_confusion_matrix,evaluate_probabilities
from ToTune.Train.Utils import print_tunning,print_point,print_evaluation_report
from IPython.display import display
def SeqCls_postprocess(SeqCls):
    labels = SeqCls.output["labels"]
    preds = SeqCls.output["preds"]
    probs = SeqCls.output["probs"] if "probs" in SeqCls.output else None

    
    if probs is not None:
        results["prob_metrics"] = evaluate_probabilities(labels=labels, probs=probs)
        results = evaluate_classification(labels=labels, predictions=preds)
    else:
        from ToTune.Train.Utils import encode_labels_and_preds
        y_true, y_pred, label2id, id2label = encode_labels_and_preds(labels, preds)
        results = evaluate_classification(labels=y_true, predictions=y_pred)
        SeqCls.output["label2id"] = label2id
        SeqCls.output["id2label"] = id2label
        SeqCls.output["y_true"] = y_true
        SeqCls.output["y_pred"] = y_pred
    SeqCls.output["evaluation"] = results
    print_evaluation_report(results)


def fullyworkflow(point):
    SeqCls = load_sequence_classification_model(point)
    prepare_data(SeqCls,point)
    trainersetting(SeqCls,point)
    SeqCls.train_test()
    SeqCls_postprocess(SeqCls)