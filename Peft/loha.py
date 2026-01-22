from peft import LoHaModel, LoHaConfig

def load_loha_model(model, **kwargs):
    kwargs['rank_dropout'] = 0.0 if 'rank_dropout' not in kwargs else kwargs['rank_dropout']
    kwargs['module_dropout'] = 0.0 if 'module_dropout' not in kwargs else kwargs['module_dropout']
    kwargs['init_weights'] = True if 'init_weights' not in kwargs else kwargs['init_weights']
    kwargs['alpha'] = 32 if 'alpha' not in kwargs else kwargs['alpha']
    config = LoHaConfig(**kwargs)
    model = LoHaModel(model, config, "default")
    kwargs['adaptation'] = 'loha'
    return model,config.to_dict()