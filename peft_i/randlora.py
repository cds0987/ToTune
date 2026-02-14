from peft import get_peft_model, RandLoraConfig
from ToTune.models.utils import is_4bit

def load_randlora_model(model, **kwargs):
    """
    Flexible  loader.
    Accepts any valid LoraConfig arguments via **lora_kwargs.
    """
    config = RandLoraConfig(**kwargs)
    model = get_peft_model(model, config)
    if is_4bit(model):
        model = model.half()
    kwargs['adaptation'] = 'randlora'
    return model,config.to_dict()