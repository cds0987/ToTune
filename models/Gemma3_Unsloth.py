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

def load_Gemma3_TextUnsloth_Model(model_name,max_seq_length,load_peft = True):
  model,tokenizer = FastModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)
  if load_peft:
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


from ToTune.models.basedModel import UnslothSFTModel
class SFTGemma3(UnslothSFTModel):
  def __init__(self,model_name,Model = None,tokenizer = None,adaptation = {},max_seq_length = 128):
    super().__init__(model_name,Model,tokenizer,adaptation,max_seq_length)
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
    self.train_onlyresponse = True
  def preprocess(self,train_ds,test_ds,text_col,label_col):
     self.train_ds = train_ds
     self.test_ds  = test_ds
     self.print_dataset()
     self.text_col = text_col
     self.label_col = label_col
     self.train_ds = preparedata(train_ds,self.tokenizer,self.instruction,text_col,label_col)
     self.test_ds  = test_ds
  def inference(self, texts: list[str]):
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(self.model)
        self.device = self.model.device
        self.model.eval()
        self.set_temperature(self.workmode)
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
                max_new_tokens = self.max_new_tokens,
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
