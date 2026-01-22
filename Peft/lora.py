from peft import get_peft_model, LoraConfig

def load_lora_model(model, **kwargs):
    """
    Flexible LoRA loader.
    Accepts any valid LoraConfig arguments via **lora_kwargs.
    """
    kwargs['lora_alpha'] = kwargs['r'] if 'lora_alpha' not in kwargs else kwargs['lora_alpha']
    kwargs['lora_dropout'] = 0.05 if 'lora_dropout' not in kwargs else kwargs['lora_dropout']
    kwargs['bias'] = "none" if 'bias' not in kwargs else kwargs['bias']
    kwargs['task_type'] = "SEQ_CLS" if 'task_type' not in kwargs else kwargs['task_type']
    config = LoraConfig(**kwargs)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    kwargs['adaptation'] = 'lora'
    return model,config.to_dict()
