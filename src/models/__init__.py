from .transformer import TransformerCaptionModel
# from .lstm import LSTMCaptionModel 

MODEL_REGISTRY = {
    "transformer": TransformerCaptionModel,
}

def build_model(config, vocab_size):
    model_name = config['model'].get('name', 'transformer')
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry")
    
    return MODEL_REGISTRY[model_name](config, vocab_size)
