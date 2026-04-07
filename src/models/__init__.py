MODEL_REGISTRY = {
    "transformer": TransformerCaptionModel,
    "lstm": LSTMCaptionModel,
    # ...
}

def build_model(config, vocab_size) -> BaseCaptionModel:
    name = config["model"]["name"]
    cls = MODEL_REGISTRY[name]
    return cls(config, vocab_size)