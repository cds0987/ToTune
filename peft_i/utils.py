from ToTune.Peft.lora import load_lora_model
from ToTune.Peft.randlora import load_randlora_model
from ToTune.Peft.ia3 import load_ia3_model
from ToTune.Peft.adalora import load_adalora_model
from ToTune.Peft.loha import load_loha_model


def load_peft_model(model,adaptation = 'lora', **kwargs):
    if adaptation == 'lora':
        return load_lora_model(model, **kwargs)
    elif adaptation == 'randlora':
        return load_randlora_model(model, **kwargs)
    elif adaptation == 'ia3':
        return load_ia3_model(model, **kwargs)
    elif adaptation == 'adalora':
        return load_adalora_model(model, **kwargs)
    elif adaptation == 'loha':
        return load_loha_model(model, **kwargs)
    else:
        pass