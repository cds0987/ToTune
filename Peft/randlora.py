from peft import get_peft_model, RandLoraConfig
from ToTune.models.Utils import is_4bit

def load_randlora_model(model, **kwargs):
    """
    Flexible  loader.
    Accepts any valid LoraConfig arguments via **lora_kwargs.
    """
    config = RandLoraConfig(**kwargs)
    model = get_peft_model(model, config)
    if is_4bit(model):
        model = model.half()
    model.print_trainable_parameters()
    kwargs['adaptation'] = 'randlora'
    return model,kwargs