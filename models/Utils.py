from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import prepare_model_for_kbit_training

def load_sequence_classification_model(
    model_name: str,
    num_labels: int,
    token: None = None,
    load_in_4bit: bool = False,
    compute_dtype: torch.dtype = torch.float16,
):
    # --- Tokenizer kwargs ---
    tok_kwargs = {}
    if token is not None:
        tok_kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)

    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        if tokenizer.pad_token == "[PAD]":
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # --- Optional 4-bit config ---
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=compute_dtype,
        )

    # --- Model kwargs ---
    model_kwargs = {
        "quantization_config": quant_config,
        "num_labels": num_labels,
        "ignore_mismatched_sizes": True,
    }
    if token is not None:
        model_kwargs["token"] = token

    # --- Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        **model_kwargs,
    )

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # Resize after adding pad token
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


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

