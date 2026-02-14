from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
def load_QwenUnsloth_Model(model_name,max_seq_length,load_peft = True):
  model,tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
  lora = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",]
  if load_peft:
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

from ToTune.models.basedModel import UnslothSFTModel
class SFTAlpacaQwen(UnslothSFTModel):
  def __init__(self,model_name,Model = None,tokenizer = None,adaptation = {},max_seq_length = 128):
    super().__init__(model_name,Model,tokenizer,adaptation,max_seq_length)
    import warnings
    warnings.filterwarnings("ignore")
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
     self.train_ds = train_ds
     self.test_ds  = test_ds
     self.print_dataset()
     self.text_col = text_col
     self.label_col = label_col
     self.train_ds = Alpacadata(train_ds,self.tokenizer,self.instruction,text_col,label_col)
     self.test_ds  = test_ds
  def inference(self, texts):
    from unsloth import FastLanguageModel  # FastVisionModel for LLMs
    FastLanguageModel.for_inference(self.model)
    self.set_temperature(self.workmode)
    batch_size = self.batch_size
    import torch
    from tqdm import tqdm

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
    import re
    preds  = [
    re.sub(r"<think>.*?</think>\s*", "", p, flags=re.DOTALL).strip()
    for p in preds
]

    return preds
