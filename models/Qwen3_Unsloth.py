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