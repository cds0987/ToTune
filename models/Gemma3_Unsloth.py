from unsloth import FastModel
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only
from unsloth.chat_templates import standardize_data_formats
from unsloth.chat_templates import get_chat_template
from tqdm import tqdm

def convert_to_chatml(example, base_prompt,text_col,label_col):
    return {
        "conversations": [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": example[text_col]},
            {"role": "assistant", "content": example[label_col]},
        ]
    }

def formatting_prompts_func(examples, tokenizer):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix("<bos>")
        for convo in convos
    ]
    return {"text": texts}

def preparedata(dataset, tokenizer, base_prompt,text_col,label_col):
    dataset = dataset.map(lambda ex: convert_to_chatml(ex, base_prompt,text_col,label_col))
    dataset = dataset.map(lambda ex: formatting_prompts_func(ex, tokenizer), batched=True)
    return dataset

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

def load_Gemma3_TextUnsloth_Model(model_name,max_seq_length):
  model,tokenizer = FastModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)
  model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 128,           # Larger = higher accuracy, but might overfit
    lora_alpha = 128,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)
  tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma3",
)
  return model,tokenizer

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
class UnslothConservationTextGemma3(BasedModel):
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
    self.adaptation = dict(target_modules = ["finetune_language_layers", "finetune_attention_modules", "finetune_mlp_modules"],
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
     self.train_ds = preparedata(train_ds,self.tokenizer,self.instruction,text_col,label_col)
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
    from unsloth.chat_templates import train_on_responses_only
    self.output['adaptation'] = self.adaptation
    self.output['Tuner_arg'] = self.extract_fields(self.essential_keys)
    try:
     self.trainer = train_on_responses_only(self.trainer,instruction_part = "<start_of_turn>user\n",response_part = "<start_of_turn>model\n",)
    except Exception as e:
     print("Error in training on responses only using default trainer:", e)
     pass
  def test(self,):
    texts = self.test_ds["meta_description"]
    preds = self.inference(texts)
    labels = self.test_ds[self.label_col]
    return preds,labels
  def train_test(self, *args, **kwargs):
    from ToTune.models.BasedModel import warn
    import torch
    warn()
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
  def inference(self, texts: list[str]):
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(self.model)
        max_new_tokens = self.max_seq_length
        self.device = self.model.device
        self.model.eval()
        preds = []

        for i in tqdm(
            range(0, len(texts), self.batch_size),
            desc="Evaluating",
            unit="batch",
        ):
            batch_texts = texts[i : i + self.batch_size]

            messages_batch = [
                [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.instruction + text,
                        }
                    ],
                }]
                for text in batch_texts
            ]

            inputs = self.tokenizer.apply_chat_template(
                messages_batch,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            outputs = self.model.generate(
                inputs,
                max_new_tokens = max_new_tokens,
                temperature = self.temperature,
                do_sample = self.temperature > 0,
                top_p = self.top_p,
                top_k = self.top_k,
                use_cache=True,
            )

            gen_tokens = outputs[:, inputs.shape[1]:]
            gen_texts = self.tokenizer.batch_decode(
                gen_tokens,
                skip_special_tokens=True,
            )

            preds.extend([t.strip() for t in gen_texts])

        return preds
