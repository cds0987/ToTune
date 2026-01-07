from transformers import TrainingArguments,Trainer
from tqdm import tqdm
from ToTune.models.BasedModel import BasedModel
class SequenceClassification(BasedModel):
  def __init__(self,model_name,max_seq_length,Model = None,tokenizer = None,target_modules = ['Not used'],r  = -1):
    super().__init__(model_name,max_seq_length,Model,tokenizer,target_modules,r)

  def preprocess(self,train_ds,test_ds,text_col,label_col):
     self.text_col = text_col
     self.label_col = label_col
     def process(batch):
         return self.tokenizer(
        batch[text_col],
        truncation=True,
        padding="max_length",
        max_length=self.max_seq_length if self.max_seq_length is not None else 128,
    )

     train_enc = train_ds.map(process, batched=True)
     train_enc = train_enc.rename_column(label_col, "labels")
     train_enc.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
     self.train_ds = train_enc
     self.test_ds  = test_ds
  def prepare_trainer(self,arg = None):
    common_args = dict(
        auto_find_batch_size = True,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1,
        learning_rate = 5e-5,
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed=3407,
        output_dir = "outputs",
        report_to = "none",
        save_strategy = "no",
        save_total_limit = 0,
    )
    if arg is not None:
        for key, value in arg.items():
            common_args[key] = value
    self.common_args =  TrainingArguments(**common_args)
    self.trainer = Trainer(
        args = self.common_args,
        model = self.model,
        train_dataset = self.train_ds
       )
    
  def inference(self, texts, max_seq_length):
    import torch
    from tqdm import tqdm
    self.max_new_tokens = max_seq_length
    batch_size = self.batch_size if hasattr(self, 'batch_size') else 32
    preds = []
    self.model.eval()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)
    for i in tqdm(range(0, len(texts), batch_size), desc="Running inference"):
        batch_texts = texts[i : i + batch_size]

        inputs = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_seq_length,
            padding_side="right"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1)

        preds.extend(batch_preds.cpu().numpy().tolist())
    return preds
