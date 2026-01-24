from ToTune.Peft.Utils import load_peft_model
from ToTune.models.Utils import load_sequence_classification_model
from ToTune.models.SequenceClassification import SequenceClassification


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


from ToTune.Train.Utils import evaluate_classification,plot_confusion_matrix,evaluate_probabilities
from ToTune.Train.Utils import print_tunning,print_point,print_evaluation_report
from IPython.display import display
def SeqCls_postprocess(SeqCls):
    print_tunning(SeqCls.output)
    SeqCls.train_test()
    labels = SeqCls.output["labels"]
    preds = SeqCls.output["preds"]
    probs = SeqCls.output["probs"]

    results = evaluate_classification(labels=labels, predictions=preds)
    results["prob_metrics"] = evaluate_probabilities(labels=labels, probs=probs)

    SeqCls.output["evaluation"] = results

    print_evaluation_report(results)


def train_SeqCls(point):
  print_point(point)
  print(f"\n===== Load Model =====")
  SeqCls = sequence_classification_model(point)
  print(f"\n===== Prepare Data =====")
  prepare_data(SeqCls,point)
  print(f"\n===== Train Model =====")
  trainersetting(SeqCls,point)
  SeqCls_postprocess(SeqCls)
  return SeqCls


def Qwen3_postprocess(SeqCls):
    print_tunning(SeqCls.output)
    SeqCls.train_test()
    labels = SeqCls.output["labels"]
    preds = SeqCls.output["preds"]
    results = evaluate_classification(labels=labels, predictions=preds)
    SeqCls.output["evaluation"] = results
    print_evaluation_report(results)
from ToTune.models.Qwen3_Unsloth import UnslothAlpacaQwen,load_Unsloth_Model

def train_UnslothQwenAlpaca(point):
  print_point(point)
  print(f"\n===== Load Model =====")
  model_name,max_seq_length = point['model_name'],point['max_seq_length']
  model,tokenizer = load_Unsloth_Model(model_name,max_seq_length)
  UnslothQwen = UnslothAlpacaQwen(model_name,model,tokenizer,max_seq_length = max_seq_length)
  UnslothQwen.instruction = point['base_prompt']
  print(f"\n===== Prepare Data =====")
  train_ds,test_ds,text_col,label_col = point['train_ds'],point['test_ds'],point['text_col'],point['label_col']
  UnslothQwen.preprocess(train_ds,test_ds,text_col,label_col)
  args = point['trainargs'] if 'trainargs' in point else None
  UnslothQwen.prepare_trainer(args)
  UnslothQwen.set_temperature(mode = point['mode'])
  print(f"\n===== Train And Test Model =====")
  Qwen3_postprocess(UnslothQwen)
  return UnslothQwen