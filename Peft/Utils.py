from ToTune.Peft.lora import load_lora_model
from ToTune.Peft.randlora import load_randlora_model

def load_peft_model(model,adaptation = 'lora', **kwargs):
    if adaptation == 'lora':
        return load_lora_model(model, **kwargs)
    elif adaptation == 'randlora':
        return load_randlora_model(model, **kwargs)