from peft import IA3Model, IA3Config

def load_ia3_model(model, **kwargs):
    kwargs['peft_type'] = "IA3"
    config = IA3Config(**kwargs)
    model = IA3Model(model, config, adapter_name="Default")
    kwargs['adaptation'] = 'ia3'
    return model,config.to_dict()
  