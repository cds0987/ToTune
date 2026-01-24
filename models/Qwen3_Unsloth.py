from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
def load_Unsloth_Model(model_name,max_seq_length):
  model,tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
  lora = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",]
  model = FastLanguageModel.get_peft_model(
    model,
    r = 128,
    target_modules = lora,
    lora_alpha = 128,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,  # And LoftQ
)
  return model,tokenizer

from unsloth import FastModel
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm
import torch
def Alpacadata(dataset,tokenizer,base_prompt,text_col,label_col):
     EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
     alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""
     def formatting_prompts_func(examples):
         instruction = base_prompt  # single string, not a list
         inputs = examples[f"{text_col}"]
         outputs = examples[f"{label_col}"]
         texts = []
         for inp, out in zip(inputs, outputs):
           text = alpaca_prompt.format(instruction, inp, out) + EOS_TOKEN
           texts.append(text)
         return { "text" : texts, }
     return dataset.map(formatting_prompts_func, batched = True,)

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

from ToTune.models.BasedModel import BasedModel
class UnslothAlpacaQwen(BasedModel):
  def __init__(self,model_name,Model = None,tokenizer = None,adaptation = {},max_seq_length = 128):
    super().__init__(model_name,Model,tokenizer,adaptation,max_seq_length)
    import warnings
    warnings.filterwarnings("ignore")
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
    self.batch_size = 8
    self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""
    self.adaptation = dict(target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
                       r = 128,
                       lora_alpha = 128,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,) if not adaptation else adaptation
  def preprocess(self,train_ds,test_ds,text_col,label_col):
     self.text_col = text_col
     self.label_col = label_col
     self.train_ds = Alpacadata(train_ds,self.tokenizer,self.instruction,text_col,label_col)
     self.test_ds  = test_ds
  def set_temperature(self,mode = 'creative'):
    if mode == 'creative':
      self.temperature = 0.7
      self.top_p = 0.8
      self.top_k = 20
    elif mode == 'determination':
      self.temperature = 0.0
      self.top_p = 1.0
      self.top_k = 0

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
    self.output['adaptation'] = self.adaptation
    self.output['Tuner_arg'] = self.extract_fields(self.essential_keys)
  def test(self,):
    texts = self.test_ds["meta_description"]
    preds = self.inference(texts)
    labels = self.test_ds[self.label_col]
    return preds,labels
  def train_test(self, *args, **kwargs):
    from ToTune.models.BasedModel import warn
    import torch
    warn()
    import warnings
    warnings.filterwarnings("ignore")
    output = {}
    from ToTune.Tools.memory import total_current_mem,total_peak_mem
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = total_current_mem()
    output['Model_name'] = self.model_name
    output['trainoutput'] = self.trainer.train()
    mem_peak = total_peak_mem()
    output['trainer'] = self.trainer
    output['FinetuneMemory'] = mem_peak - mem_before
    torch.cuda.empty_cache()
    output['Train_size'] = len(self.train_ds)
    output['Test_size'] = len(self.test_ds)
    preds,labels = self.test()
    y_true, y_pred, label2id, id2label = encode_labels_and_preds(labels, preds)

    output['preds'] = y_pred
    output['labels'] = y_true
    output['label2id'] = label2id
    output['id2label'] = id2label
    output['Tuner_arg'] = self.extract_fields(self.essential_keys)
    output['adaptation'] = self.adaptation
    self.output = output
  def inference(self, texts):
    from unsloth import FastLanguageModel  # FastVisionModel for LLMs
    batch_size = self.batch_size
    max_seq_length = self.max_seq_length
    import torch
    from tqdm import tqdm

    self.max_new_tokens = max_seq_length
    preds = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model.eval()
    self.model.to(device)

    for i in tqdm(
        range(0, len(texts), batch_size),
        desc="Evaluating",
        unit="batch"
    ):
        batch_texts = texts[i : i + batch_size]

        # 1. Build prompts
        messages_batch = []
        for text in batch_texts:
            prompt_text = (
                self.alpaca_prompt.format(self.instruction, text, "")
                + self.tokenizer.eos_token
            )
            messages_batch.append(
                [{"role": "user", "content": prompt_text}]
            )

        # 2. Apply chat template (BATCHED)
        inputs = self.tokenizer.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # 3. Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens = self.max_new_tokens,
                temperature = self.temperature,
                top_p = self.top_p,
                top_k = self.top_k,
                do_sample = self.temperature > 0,
                use_cache=True,
            )

        # 4. Decode only generated tokens
        gen_tokens = outputs[:, inputs.shape[1]:]
        gen_texts = self.tokenizer.batch_decode(
            gen_tokens,
            skip_special_tokens=True
        )

        preds.extend([t.strip() for t in gen_texts])

    return preds
