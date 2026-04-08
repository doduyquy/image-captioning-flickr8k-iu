import torch
import torch.nn as nn
from ..components.positional_encoding import TokenPositionalEmbedding

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, ff_dim=2048, max_len=512):
        super().__init__()
        # Đối với text, cần có cả Embedding từ điển và Vị trí
        self.embedding = TokenPositionalEmbedding(vocab_size, embed_dim, max_len)
        
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, encoder_out, return_attention=False):
        """
        x: [B, T] (Cụm từ đã sinh ra)
        encoder_out: [B, 49, 512] (Đặc trưng hình ảnh từ Encoder)
        """
        x = self.embedding(x)
        
        # Causal mask giúp decoder không "nhìn trộm" tương lai
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        
        # 1. Self Attention giữa các từ trong câu đang sinh
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, attn_mask=mask)
        x = x + attn_out
        
        # 2. Cross Attention: Decoder hỏi Encoder về đặc trưng ảnh
        norm_x = self.norm2(x)
        attn_out, attn_weights = self.cross_attn(norm_x, encoder_out, encoder_out, average_attn_weights=False)
        x = x + attn_out
        
        # 3. Feedforward
        ff_out = self.ff(self.norm3(x))
        x = x + ff_out
        
        logits = self.fc_out(x)
        
        if return_attention:
            return logits, attn_weights
        return logits
