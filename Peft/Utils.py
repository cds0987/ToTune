from ToTune.Peft.lora import load_lora_model
from ToTune.Peft.randlora import load_randlora_model

def load_peft_model(model, **kwargs):
    if kwargs['adaptation'] == 'lora':
        return load_lora_model(model, **kwargs)
    elif kwargs['adaptation'] == 'randlora':
        return load_randlora_model(model, **kwargs)