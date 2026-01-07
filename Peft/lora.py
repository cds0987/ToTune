from peft import get_peft_model, LoraConfig

def load_lora_model(model, **kwargs):
    """
    Flexible LoRA loader.
    Accepts any valid LoraConfig arguments via **lora_kwargs.
    """
    config = LoraConfig(**kwargs)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    kwargs['adaptation'] = 'lora'
    return model,kwargs
