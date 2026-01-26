import torch
from sentence_transformers import SentenceTransformer
import gc

def encode(model_name, texts,max_seq_length = 128,batch_size = 4, savedir=None, name="_IndustryEbd"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    model = model.to(torch.bfloat16)

    # 🔥 Set manual sequence length
    model.max_seq_length = max_seq_length
    if hasattr(model, "tokenizer"):
        model.tokenizer.model_max_length = max_seq_length
        model.tokenizer.init_kwargs["model_max_length"] = max_seq_length

    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        device=device,
        batch_size=batch_size,
        show_progress_bar=True
    )

    embeddings = embeddings.cpu()

    savename = model_name.replace('/', '-') if savedir is not None else None
    if savename is not None:
        torch.save(embeddings, f'{savedir}/{savename}.pt')
        print(f"Saved embeddings to {savedir}/{savename}.pt")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return embeddings

from tqdm import tqdm
from ToTune.models.BasedModel import BasedModel
class EmbeddingsClassification(BasedModel):
    def __init__(self,model_name,machinelearning_name,MachineLearning = None,Model = None,tokenizer = None,adaptation = {},max_seq_length = 128):
        super().__init__(model_name,Model,tokenizer,adaptation,max_seq_length)
        self.machinelearning_name = machinelearning_name
        self.MachineLearning = MachineLearning
    def preprocess(self,train_ds,test_ds,text_col,label_col):
        self.text_col = text_col
        self.label_col = label_col
        self.train_texts = train_ds[self.text_col]
        self.test_texts  = test_ds[self.text_col]
        self.train_labels = train_ds[self.label_col]
        self.test_labels  = test_ds[self.label_col]
        all_texts = self.train_texts + self.test_texts
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataemb = self.Model.encode(
            all_texts,
            convert_to_tensor=True,
            batch_size = self.batch_size,
            device=self.device,
            show_progress_bar=True
        ).cpu().numpy()
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
        embeddings = self.Model.encode(
            texts,
            convert_to_tensor=True,
            batch_size = self.batch_size,
            device=self.device,
            show_progress_bar=True
        ).cpu().numpy()  
        return self.MachineLearning.predict(embeddings)

    def test(self,):
        texts = self.test_texts
        preds = self.inference(texts)
        labels = self.test_labels
        self.output['preds'] = preds
        self.output['labels'] = labels 
