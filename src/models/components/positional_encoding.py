import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    Chỉ thực hiện cộng thêm vector vị trí vào đầu vào (thường dùng cho Visual Features).
    """
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        
        # tạo ma trận PE [max_len, embed_dim]
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even index
        pe[:, 1::2] = torch.cos(position * div_term)  # odd index

        pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]

        # không huấn luyện (không train)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        T = x.shape[1]
        return x + self.pe[:, :T, :]


class TokenPositionalEmbedding(nn.Module):
    """
    Kết hợp cả Embedding từ điển và Positional Encoding (dành cho Text/Decoder).
    """
    def __init__(self, vocab_size, embed_dim, max_len=5000):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pe = SinusoidalPositionalEncoding(embed_dim, max_len)

    def forward(self, x):
        """
        x: [B, T] (Indices)
        """
        x = self.token_embed(x) # [B, T, D]
        return self.pe(x)