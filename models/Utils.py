import torch

def get_lora_modules(model,type = 'att'):
    target_modules = set()
    ffd_modules = set()
    attention_modules = set()

    # Detect all supported linear classes (Linear, Linear4bit, Linear8bitLt)
    try:
        import bitsandbytes as bnb
        linear_classes = (torch.nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except ImportError:
        linear_classes = (torch.nn.Linear,)

    # ---- Pass 1: Identify all linear layers in order ----
    linear_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, linear_classes):
            linear_layer_names.append(name)

    # If no linear layers found, return empty sets
    if not linear_layer_names:
        return [], [], []

    # The last layer is simply the last linear module in traversal order
    last_linear_layer = linear_layer_names[-1].lower()

    # ---- Pass 2: Classify layers + exclude last layer ----
    for name, module in model.named_modules():
        lname = name.lower()

        # Skip the last linear layer dynamically
        if lname == last_linear_layer:
            continue

        if isinstance(module, linear_classes):
            sub = name.split(".")[-1]

            # Record all linear layers
            target_modules.add(sub)

            # Attention: q, k, v layers
            if any(att in lname for att in ["q", "k", "v"]):
                attention_modules.add(sub)
            else:
                ffd_modules.add(sub)
    if type == 'all':
        modules = sorted(target_modules)
    elif type == 'att':
        modules = sorted(attention_modules)
    else:
        modules = sorted(ffd_modules)
    return modules


def is_4bit(model):
    for module in model.modules():
        if "4bit" in module.__class__.__name__.lower():
            return True
    return False
