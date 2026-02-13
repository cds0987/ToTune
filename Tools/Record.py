from datetime import datetime
from zoneinfo import ZoneInfo

def get_current_datetime(zone = "Asia/Ho_Chi_Minh"):
    now = datetime.now(ZoneInfo(zone))
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "hour": now.hour,
        "minute": now.minute,
        "second": now.second
    }
    
def dataset_agreegate(point):
    point['train_numsamples'] = len(point['train_ds'])
    point['test_numsamples'] = len(point['test_ds'])


       
       
def _record(point):
  model_load = {}
  model_load['model_name'] = point['model_name']
  model_load['max_seq'] = point['max_seq']
  model_load['use_gradient_checkpointing'] = point['use_gradient_checkpointing']
  model_load['load_in_4bit'] = point['load_in_4bit']
  model_load['num_labels'] = point['num_labels']
  model_load['tunemode'] = point['tunemode']
  dataset = {}
  dataset['dataname'] = point['dataname']
  dataset['texts_col'] = point['texts_col']
  dataset['labels_col'] = point['labels_col']
  dataset['train_numsamples'] = point['train_numsamples']
  dataset['test_numsamples']  = point['test_numsamples']
  return model_load,dataset,point['peft_config'],point['trainargs'],point['train_report'],point['evaluation'],point['timestamp']
import pandas as pd

def record_to_dataframe(point):
    model_load, dataset, peft_config, trainargs, train_report, evaluation, timestamp = _record(point)

    record = {
        "model_load": model_load,
        "dataset": dataset,
        "peft_config": peft_config,
        "trainargs": trainargs,
        "train_report": train_report,
        "evaluation": evaluation,
        "timestamp": timestamp,
    }

    return pd.DataFrame([record])


