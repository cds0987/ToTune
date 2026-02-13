def save_ModelHgface(self, model, tokenizer, username, model_name, token):
    model.model.push_to_hub(f"{username}/{model_name}", token=token, private=False)
    tokenizer.push_to_hub(f"{username}/{model_name}", token=token, private=False)
    