import torch
import torch.nn as nn

class BaseCaptionModel(nn.Module):
    def __init__(self, encoder=None, decoder=None):
        super().__init__()
        self.encoder_module = encoder
        self.decoder_module = decoder

    def forward(self, images, captions):
        """
        Nhiệm vụ: Nhận ảnh và captions, trả về logits.
        Cần được override ở lớp con.
        """
        raise NotImplementedError

    def generate(self, image, vocab, max_len, strategy, beam_size):
        """
        Nhiệm vụ: Sinh caption từ ảnh.
        Cần được override ở lớp con.
        """
        raise NotImplementedError