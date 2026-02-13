import torch
from sentence_transformers import SentenceTransformer
import gc

def LoadEmbeddingModel(model_name,max_seq_length = 128):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    model = model.to(torch.bfloat16)

    # 🔥 Set manual sequence length
    model.max_seq_length = max_seq_length
    if hasattr(model, "tokenizer"):
        model.tokenizer.model_max_length = max_seq_length
        model.tokenizer.init_kwargs["model_max_length"] = max_seq_length
    return model,model.tokenizer
from ToTune.TrainTextCls.Utils import print_tunning,print_point,print_evaluation_report
from ToTune.models.BasedModel import warn
import torch
warn()
from ToTune.Tools.memory import total_current_mem,total_peak_mem
from tqdm import tqdm
from ToTune.models.BasedModel import BasedModel
import time
class EmbeddingsClassification(BasedModel):
    def __init__(self,model_name,machinelearning_name,machinelearning_model = None,Model = None,tokenizer = None,adaptation = {},max_seq_length = 128):
        super().__init__(model_name,Model,tokenizer,adaptation,max_seq_length)
        self.machinelearning_name = machinelearning_name
        self.machinelearning_model = machinelearning_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 8
    def preprocess(self,train_ds,test_ds,text_col,label_col):
        self.train_ds = train_ds
        self.test_ds  = test_ds
        self.print_dataset()
        self.text_col = text_col
        self.label_col = label_col
        self.train_texts = train_ds[self.text_col]
        self.test_texts  = test_ds[self.text_col]
        self.train_labels = train_ds[self.label_col]
        self.test_labels  = test_ds[self.label_col]
        all_texts = self.train_texts + self.test_texts
        self.output['train_report'] = {} 
        mem_before = total_current_mem()
        start_time = time.time()
        self.dataemb = self.model.encode(
            all_texts,
            convert_to_tensor=True,
            batch_size = self.batch_size,
            device=self.device,
            show_progress_bar=True
        )
        mem_peak = total_peak_mem()
        end_time = time.time()
        self.output['train_report']['FinetuneMemory'] =  f"{round((mem_peak - mem_before) / 1024, 2)} (GB)"
        self.output['train_report']['FinetuneTime'] = end_time - start_time
        self.dataemb = self.dataemb.float().cpu().numpy()
        self.train_emb = self.dataemb[:len(self.train_texts)]
        self.test_emb  = self.dataemb[len(self.train_texts):]
    def train(self,):
        from ToTune.models.BasedModel import warn
        warn()
        import time
        start_time = time.time()
        self.machinelearning_model.fit(self.train_emb, self.train_labels)
        self.output['machinelearning_model'] = self.machinelearning_model
        self.output['MachineLearning_name'] = self.machinelearning_name
        end_time = time.time()
        self.output['FinetuneTime'] = self.output['FinetuneTime']  + (end_time - start_time) if 'FinetuneTime' in self.output else end_time - start_time
        self.output['FinetuneTime'] = f"{round(self.output['FinetuneTime'] / 3600, 4)} (Hrs)"
        self.output['train_report']['FinetuneTime'] = self.output['FinetuneTime']
        print_point(self.output['train_report'], "Tuner Tracking")
    def inference(self, texts):
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            batch_size = self.batch_size,
            device=self.device,
            show_progress_bar=True
        )
        embeddings = embeddings.float().cpu().numpy()
        return self.machinelearning_model.predict(embeddings)

    def test(self,):
        texts = self.test_texts
        preds = self.inference(texts)
        labels = self.test_labels
        self.output['preds'] = preds
        self.output['labels'] = labels
    def train_test(self, *args, **kwargs):
       self.train()
       print(f"\n===== Test Model =====")
       self.test()   

