import torch 
import torch.nn as nn
import torchvision.models as models


# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super().__init__()
#         hidden = max(in_channels // reduction, 1)

#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.mlp = nn.Sequential(
#             nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.mlp(self.avg_pool(x))
#         max_out = self.mlp(self.max_pool(x))
#         attn = self.sigmoid(avg_out + max_out)
#         return x * attn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(in_channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x)) # hiểu tổng thể
        max_out = self.mlp(self.max_pool(x)) # hiểu điểm nổi bật
        attn = self.sigmoid(avg_out + max_out) # cộng lại
        return x * attn # nhân vào để điều chỉnh độ quan trọng của model

class SpatialAttention(nn.Module):
    def __init__(self, kernel=7):
        super().__init__()
        assert kernel in (3, 7)
        padding = 3 if kernel == 7 else 1
        self.conv1=nn.Conv2d(2, 1, kernel_size=kernel, padding=padding, bias=False)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        avg_out=torch.mean(x,dim=1,keepdim=True) # tinhs trng bình của tất feature tại vị trí [i,j] của 512 chiều
        max_out=torch.max(x,dim=1,keepdim=True)[0] # tính max của tất feature tại vị trí [i,j] của 512 chiều
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv1(attn))# ghép lại thành 2 kênh, sau đó đưa về conv1 rồi lấy sigmoid để tính xác xuất
        return x*attn #ouput[B,512,7,7], chỗ này tạo 512d cái ma trận của spatial 
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x




"""
--input: [B,512,7,7]
--output: [B,512,7,7]

trong đó: channelattention: học xem trong 512d này thì kênh nào quan trọng
và spatialattention: học xem trong 7x7 này thì vị trí nào quan trọng
từ đó nhân vào để điều chỉnh độ quan trọng của model
đầu tiên là vào channel attention thì ta biết được trong 512 kênh thì kênh nào qutrong rồi
sau đó qua spatial để học không gian nữa. 
"""
class CNNEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        efficientnet=models.efficientnet_b0(pretrained=True)
        self.backbone=efficientnet.features # bo classfier
        self.conv=nn.Conv2d(1280, embed_dim, kernel_size=1)
        self.cbam = CBAM(embed_dim)
    def forward(self,images):
        features=self.backbone(images) #(B, 1280,7,7)
        features=self.conv(features) #(B, emb_dim(512), 7,7) -> giảm chiều dữ liệu, không dùng dense
        features=self.cbam(features) #(B, emb_dim(512), 7,7) cbam
        features=features.flatten(2) #(B, 512,49) trải phẳng
        features=features.permute(0,2,1)#(B,49,512) đổi chỗ 
        return features


        #49 là số lượng patch (7x7) -> image patch features
        #512 là số chiều của mỗi patch
if __name__ == "__main__":
    # Test nhanh CNNEncoder
    cnn_encoder = CNNEncoder(embed_dim=512)
    images = torch.randn(2, 3, 224, 224)
    features = cnn_encoder(images)
    print("Features shape:", features.shape) # Expected: [2, 49, 512]
    #-->  torch.Size([2, 49, 512])