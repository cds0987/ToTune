import time
import torch
import pandas as pd
import gc
import warnings
def warn():
    warnings.filterwarnings(
    "ignore",
    message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly"
)
class BasedModel:
    def __init__(self, model_name,model = None,tokenizer = None,adaptation = {}, max_seq_length = 128):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.adaptation = adaptation
        self.load_model(model,tokenizer)
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
       pass
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
       pass






