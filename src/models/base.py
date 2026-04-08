import torch
import torch.nn as nn

class BaseCaptionModel(nn.Module):
    def __init__(self, encoder=None, decoder=None):
        super().__init__()
        self.encoder_module = encoder
        self.decoder_module = decoder

    def encoder(images) -> torch.Tensor: #-> features
        pass
    def decoder(captions, features) -> torch.Tensor: # -> logits
        pass

    def forward(images, captions):
        # encoder -> decoder
        pass

    def generate(image, vocab, max_len, stratergy, beam_size) -> str: # caption (Q thay mot so doc goi la: candidate)
        pass