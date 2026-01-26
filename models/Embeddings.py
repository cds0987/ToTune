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

from tqdm import tqdm
from ToTune.models.BasedModel import BasedModel
class EmbeddingsClassification(BasedModel):
    def __init__(self,model_name,machinelearning_name,MachineLearning = None,Model = None,tokenizer = None,adaptation = {},max_seq_length = 128):
        super().__init__(model_name,Model,tokenizer,adaptation,max_seq_length)
        self.machinelearning_name = machinelearning_name
        self.MachineLearning = MachineLearning
        self.batch_size = 8
    def preprocess(self,train_ds,test_ds,text_col,label_col):
        self.text_col = text_col
        self.label_col = label_col
        self.train_texts = train_ds[self.text_col]
        self.test_texts  = test_ds[self.text_col]
        self.train_labels = train_ds[self.label_col]
        self.test_labels  = test_ds[self.label_col]
        all_texts = self.train_texts + self.test_texts
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataemb = self.model.encode(
            all_texts,
            convert_to_tensor=True,
            batch_size = self.batch_size,
            device=self.device,
            show_progress_bar=True
        )
        self.dataemb = self.dataemb.float().cpu().numpy()
        self.train_emb = self.dataemb[:len(self.train_texts)]
        self.test_emb  = self.dataemb[len(self.train_texts):]
    def train(self,):
        from ToTune.models.BasedModel import warn
        warn()
        import time
        start_time = time.time()
        self.MachineLearning.fit(self.train_emb, self.train_labels)
        self.output['MachineLearning'] = self.MachineLearning
        self.output['MachineLearning_name'] = self.machinelearning_name
        end_time = time.time()
        self.output['FinetuneTime'] = end_time - start_time
    def inference(self, texts):
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            batch_size = self.batch_size,
            device=self.device,
            show_progress_bar=True
        )
        embeddings = embeddings.float().cpu().numpy()
        return self.MachineLearning.predict(embeddings)

    def test(self,):
        texts = self.test_texts
        preds = self.inference(texts)
        labels = self.test_labels
        self.output['preds'] = preds
        self.output['labels'] = labels
    def train_test(self, *args, **kwargs):
       print(f"\n===== Train Model =====")
       self.train()
       print(f"\n===== Test Model =====")
       self.test()   
