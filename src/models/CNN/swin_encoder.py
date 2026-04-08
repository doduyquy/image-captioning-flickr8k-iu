import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights

class SwinEncoder(nn.Module):
    """
    Swin Transformer Encoder (Tiny)
    Trích xuất đặc trưng spatial từ ảnh và giảm số chiều về embed_dim (512).
    """
    def __init__(self, embed_dim=512, pretrained=True):
        super().__init__()
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        self.swin = swin_t(weights=weights)
        self.backbone = self.swin.features
        self.projection = nn.Linear(768, embed_dim)
        
    def forward(self, images):
        """
        Args:
            images: [B, 3, 224, 224]
        Returns:
            features: [B, 49, embed_dim] (Kích thước 7x7 = 49 patches)
        """
        # Feature extraction: [B, H, W, C] (Swin thường trả về channel-last)
        x = self.backbone(images)
        
        if x.dim() == 4:
            # Kiểm tra xem là [B, C, H, W] hay [B, H, W, C]
            if x.shape[1] == 768: # Channel-first
                # [B, 768, 7, 7] -> [B, 768, 49] -> [B, 49, 768]
                B, C, H, W = x.shape
                x = x.view(B, C, H * W).permute(0, 2, 1)
            else: # Channel-last [B, 7, 7, 768]
                # [B, 7, 7, 768] -> [B, 49, 768]
                B, H, W, C = x.shape
                x = x.view(B, H * W, C)
        
        # Giảm chiều xuống embed_dim (512)
        features = self.projection(x)
        
        return features

if __name__ == "__main__":
    # Test thử Encoder
    model = SwinEncoder(embed_dim=512)
    mock_img = torch.randn(2, 3, 224, 224)
    output = model(mock_img)
    print(f"Input shape: {mock_img.shape}")
    print(f"Output shape: {output.shape}") # Expect: [2, 49, 512]
