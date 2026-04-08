import torch
import torch.nn as nn
from .cnn_encoder_fusion import CNNEncoderFusion
from .swin_encoder import SwinEncoder

class HybridEncoder(nn.Module):
    """
    Hybrid Encoder kết hợp sức mạnh của CNN (EfficientNet + Attention) 
    và Swin Transformer để trích xuất đặc trưng hình ảnh tốt nhất.
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        
        # 1. Khởi tạo 2 bộ Encoder thành phần
        self.cnn_branch = CNNEncoderFusion(embed_dim=embed_dim)
        self.swin_branch = SwinEncoder(embed_dim=embed_dim)
        
        # 2. Lớp Fusion (Ghép nối và Chiếu)
        # Vì mỗi nhánh trả về embed_dim (512), khi concat sẽ thành 1024
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU() 
        )
        
    def forward(self, images):
        """
        Args:
            images: [B, 3, 224, 224]
        Returns:
            features: [B, 49, embed_dim] (đã được fusion)
        """
        # Nhánh 1: CNN (EfficientNet + CBAM)
        feat_cnn = self.cnn_branch(images)    # [B, 49, 512]
        
        # Nhánh 2: Vision Transformer (Swin-T)
        feat_swin = self.swin_branch(images)  # [B, 49, 512]
        
        # Kết hợp theo chiều vector đặc trưng (Channel dimension)
        combined = torch.cat([feat_cnn, feat_swin], dim=-1) # [B, 49, 1024]
        
        # Nén lại về 512 chiều
        features = self.fusion_layer(combined) # [B, 49, 512]
        
        return features

if __name__ == "__main__":
    # Test thử Hybrid Encoder
    model = HybridEncoder(embed_dim=512)
    mock_img = torch.randn(2, 3, 224, 224)
    output = model(mock_img)
    print(f"Hybrid Input shape: {mock_img.shape}")
    print(f"Hybrid Output shape: {output.shape}") # Expect: [2, 49, 512]
    
    # Kiểm tra xem có nan không (đề phòng LayerNorm/SiLU)
    if not torch.isnan(output).any():
        print("Success: Không có giá trị NaN trong kết quả Fusion.")
