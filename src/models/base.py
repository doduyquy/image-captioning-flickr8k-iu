import torch
import torch.nn as nn

class BaseCaptionModel(nn.Module):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def encoder(images) -> torch.Tensor: #-> features
        pass
    def decoder(captions, features) -> torch.Tensor: # -> logits
        pass

    def forward(images, captions):
        # encoder -> decoder
        pass

    def generate(image, vocab, max_len, stratergy, beam_size) -> str: # caption (Q thay mot so doc goi la: candidate)
        pass