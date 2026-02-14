from peft import LoraConfig, AdaLoraModel, AdaLoraConfig

def load_adalora_model(model, **kwargs):
    kwargs['peft_type'] = "ADALORA"
    config = AdaLoraConfig(**kwargs)
    model = AdaLoraModel(model, config, "default")
    kwargs['adaptation'] = 'adalora'
    return model,config.to_dict()