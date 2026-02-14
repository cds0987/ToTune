import pandas as pd
from huggingface_hub import login
from datasets import Dataset, concatenate_datasets,load_dataset

def pushdata_to_hgface(data_name,df,token):
   dataset = Dataset.from_pandas(df)
   dataset.push_to_hub(f"{data_name}", private=False, token=token)
   
from huggingface_hub import login
def hf_login(point):
    if point.get('token', None) is not None:
       login(point['token'])
       return True
    else:
       print("No Huggingface token provided, skipping login.")
       return False