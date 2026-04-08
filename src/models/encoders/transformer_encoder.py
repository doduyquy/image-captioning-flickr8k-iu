import torch
import torch.nn as nn
from ..components.positional_encoding import SinusoidalPositionalEncoding

class TransformerEncoder(nn.Module):
    """
    Encoder xử lý các đặc trưng không gian (spatial features) từ CNN.
    """
    def __init__(self, embed_dim=512, num_heads=8, max_len=5000):
        super().__init__()
        self.position_encoding = SinusoidalPositionalEncoding(embed_dim, max_len)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, embed_dim)
        )

    def forward(self, x):
        """
        x: [B, T, D] (v.d: [B, 49, 512]) từ CNN Encoder
        """
        # Cộng thêm thông tin vị trí cho các vùng ảnh
        x = self.position_encoding(x)
        
        # Attention block (Self-attention giữa các vùng ảnh)
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # Feedforward block
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out

        return x
if __name__ == "__main__": 
    transformer_encoder=TransformerEncoder(embed_dim=512, num_heads=8, max_len=5000)
    x=torch.randn(2,49,512)
    features=transformer_encoder(x)
    print("Features shape:", features.shape) # Expected: [2, 49, 512]
    #--> torch.Size([2, 49, 512])