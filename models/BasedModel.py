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
        self.output = {}
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
    def inference(self, texts):
        pass
    def save_modelHgface(self, *args, **kwargs):
        pass
    def test(self,*args, **kwargs):
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
       from ToTune.trainTextCls.utils import print_tunning,print_point,print_evaluation_report
       self.output['adaptation'] = self.adaptation
       self.output['Tuner_arg'] = self.extract_fields(self.essential_keys)
       print_tunning(self.output)
       print(f"\n===== Train Model =====")
       self.train()
       print(f"\n===== Test Model =====")
       self.test()
    def print_dataset(self,):
       from ToTune.trainTextCls.utils import print_dataset_demo
       print_dataset_demo([self.train_ds,self.test_ds])
    def train(self,):
        from ToTune.trainTextCls.utils import print_tunning,print_point,print_evaluation_report
        import torch
        warn()
        from ToTune.tools.memory import total_current_mem,total_peak_mem
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = total_current_mem()
        import time
        self.output['train_report'] = {}
        self.output['Model_name'] = self.model_name
        start_time = time.time()
        self.output['trainoutput'] = self.trainer.train()
        mem_peak = total_peak_mem()
        self.output['trainer'] = self.trainer
        self.output['train_report']['FinetuneMemory'] =  f"{round((mem_peak - mem_before) / 1024, 2)} (GB)"
        end_time = time.time()
        self.output['train_report']['FinetuneTime'] = f"{round((end_time - start_time) / 3600, 2)} (Hrs)"
        trainable_parameters, total_parameters = self.count_paramaters()
        self.output['train_report']['FinetuneParameters'] = trainable_parameters
        self.output['train_report']['TotalParameters'] = total_parameters
        self.output['train_report']['PercentFinetuneParameters'] = f'{trainable_parameters / total_parameters * 100:.2f}%'
        print_point(self.output['train_report'], "Tuner Tracking")







class UnslothSFTModel(BasedModel):
    def __init__(self,model_name,Model = None,tokenizer = None,adaptation = {},max_seq_length = 128):
        super().__init__(model_name,Model,tokenizer,adaptation,max_seq_length)
        self.train_onlyresponse = False
        self.workmode = 'creative'
    def set_temperature(self,mode = 'creative'):
     if mode == 'creative':
      self.temperature = 0.7
      self.top_p = 0.8
      self.top_k = 20
     elif mode == 'determination':
      self.temperature = 0.0
      self.top_p = 1.0
      self.top_k = 0
    def set_train_onlyresponse(self,):
       from unsloth.chat_templates import train_on_responses_only
       if self.train_onlyresponse:
          try:
             self.trainer = train_on_responses_only(self.trainer,instruction_part = "<start_of_turn>user\n",response_part = "<start_of_turn>model\n",)
          except Exception as e:
            print("Error in training on responses only using default trainer:", e)
            pass
       else:
          pass
    def prepare_trainer(self,arg = None, mode="work"):
        from trl import SFTConfig, SFTTrainer
        default_args = {
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 1,
        "warmup_steps": 5,
        "learning_rate": 2e-5,
        "logging_steps": 10,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "output_dir": "outputs",
        "report_to": "none",
    }
        if arg:
          default_args.update(arg)
        if mode == "demo":
           default_args["max_steps"] = 3  # run only 3 training steps
           print("⚙️ Running in DEMO mode (max_steps=3)")
        else:
           print("🚀 Running in WORK mode (full training)")
        self.common_args = SFTConfig(
         **default_args
        )
        self.trainer = SFTTrainer(
           model = self.model,
           tokenizer = self.tokenizer,
           train_dataset = self.train_ds,
           dataset_text_field = "text",
           max_seq_length = self.max_seq_length,
           args = SFTConfig(
           **default_args
        ),
)
        self.set_train_onlyresponse()
    def test(self,):
      texts = self.test_ds[self.text_col]
      preds = self.inference(texts)
      labels = self.test_ds[self.label_col]
      self.output['preds'] = preds
      self.output['labels'] = labels
    