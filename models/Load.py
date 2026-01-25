from ToTune.models.Qwen3_Unsloth import load_QwenUnsloth_Model
from ToTune.models.Gemma3_Unsloth import SFTGemma3
from ToTune.models.Utils import load_sequence_classification_model
from ToTune.models.SequenceClassification import SequenceClassification
from ToTune.Peft.Utils import load_peft_model


def Load_sequence_classification_model(point):
    load_in_4bit = point['load_in_4bit']
    num_labels = point['num_labels']
    use_gradient_checkpointing = point['use_gradient_checkpointing']
    model_name = point['model_name']
    max_seq = point['max_seq']
    Used_model, tokenizer = load_sequence_classification_model(
        model_name, num_labels, load_in_4bit=load_in_4bit,use_gradient_checkpointing = use_gradient_checkpointing
    )
    peft_config = point['peft_config'] if 'peft_config' in point else None
    if peft_config is None:
        peft_config = {}
        peft_config['target_modules'] = ['Not used']
        peft_config['r'] = -1
    else:
        adaptation = peft_config['adaptation']
        peft_config.pop("adaptation", None)
        Used_model,adaptation = load_peft_model(Used_model,adaptation = adaptation, **peft_config)
        peft_config = adaptation
    SeqCls = SequenceClassification(model_name,Used_model,tokenizer,peft_config,max_seq)
    return SeqCls
from ToTune.models.Gemma3_Unsloth import load_Gemma3_TextUnsloth_Model
def loadSFTgemma3(point):
    model_name = point['model_name']
    max_seq_length = point['max_seq_length']
    model,tokenizer = load_Gemma3_TextUnsloth_Model(model_name,max_seq_length)
    Gemma3 = SFTGemma3(model_name,model,tokenizer,max_seq_length = max_seq_length)
    instruction = point['instruction']
    Gemma3.instruction = instruction
    return Gemma3
from ToTune.models.Qwen3_Unsloth import load_QwenUnsloth_Model
from ToTune.models.Qwen3_Unsloth import SFTAlpacaQwen
def loadSFTQwen3Alpaca(point):
    model_name = point['model_name']
    max_seq_length = point['max_seq_length']
    model,tokenizer = load_QwenUnsloth_Model(model_name,max_seq_length)
    Qwen3 = SFTAlpacaQwen(model_name,model,tokenizer,max_seq_length = max_seq_length)
    instruction = point['instruction']
    Qwen3.instruction = instruction
    return Qwen3



