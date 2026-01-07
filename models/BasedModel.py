import time
import torch
import pandas as pd
import gc
class BasedModel:
    def __init__(self, model_name,model = None,tokenizer = None,adaptation = {}, max_seq_length = 128):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.adaptation = adaptation
        self.load_model(model,tokenizer)
        self.essential_keys = [

    # --- Data / Batching ---
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",

    # --- Optimization ---
    "learning_rate",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "max_grad_norm",
    "optim",
    "optim_args",
    "adafactor",

    # --- Training schedule ---
    "num_train_epochs",
    "max_steps",
    "lr_scheduler_type",
    "warmup_steps",
    "warmup_ratio",

    # --- Precision ---
    "fp16",
    "bf16",
    "fp16_opt_level",
    "half_precision_backend",

    # --- Misc ---
    "label_smoothing_factor",
]
    def load_model(self,Model = None,tokenizer = None):
      if Model  is  None:
        raise ValueError("You must provide model and tokenizer explicitly")
      else:
         self.model = Model
         self.tokenizer = tokenizer
    def preprocess(self, *args, **kwargs):
        pass
    def prepare_trainer(self, *args, **kwargs):
        pass
    def inference(self, text):
        pass
    def save_modelHgface(self, *args, **kwargs):
        pass
    def test(self,max_newtokens):
        texts = self.test_ds[self.text_col]
        preds = self.inference(texts,max_newtokens)
        labels = self.test_ds[self.label_col]
        return preds,labels[:len(preds)]
    def clear_memory(self, *args, **kwargs):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    def count_paramaters(self):
        try:
            trainable_params, total_params = self.model.get_nb_trainable_parameters()
        except AttributeError:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return trainable_params, total_params
    def extract_fields(self,keys: list, missing_value="Not have"):
     source = self.common_args.to_dict()
     result = {}
     for k in keys:
        v = source.get(k, None)
        result[k] = v if v is not None else missing_value
     result['n_gpu'] = self.trainer.args.n_gpu
     return result
    def train_test(self, *args, **kwargs):
        output = {}
        output['Model_name'] = self.model_name
        output['train'] = self.trainer.train()
        output['Train_size'] = len(self.train_ds)
        output['Test_size'] = len(self.test_ds)
        output['preds'] = preds
        output['labels'] = labels
        preds,labels = self.test(self.max_seq_length)
        output['Tuner_arg'] = self.extract_fields(self.essential_keys)
        output['adaptation'] = self.adaptation
        self.output = output






