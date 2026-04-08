import torch
import torch.nn as nn
import torchvision.models as models


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



class CNNEncoderFusion(nn.Module):
    def __init__(self, embed_dim=512, refine=True):
        super().__init__()
        self.refine = refine
        weights = models.EfficientNet_B0_Weights.DEFAULT
        efficientnet=models.efficientnet_b0(weights=weights)
        features=efficientnet.features

        # ta cos các thành phần
        #input [224x224x3]
        #features[0]: conv [112x112x32]
        #features[1]: Mb con v1 trong MB có expand (nhân lên) --> depth(học spatial từng kêh) --> projection(nén lại) [112x112x16]
        #features[2]: Mb con v2 [56x56x24]
        #features[3]: Mb con v3 [28x28x40]
        #features[4]: Mb con v4 [14x14x80]
        #features[5]: Mb con v5 [14x14x112]
        #features[6]: Mb con v6 [7x7x192]
        #features[7]: Mb con v7 [7x7x320]
        #features[8]: conv 1x1 cuối 

        self.stage0_5=nn.Sequential(*features[:6]) # thuc thi den het MB CONV5
        self.stage6=nn.Sequential(features[6]) # mbv6
        self.stage7=nn.Sequential(features[7])#mbv7
        #fusion lại giữa 2 stage
        self.fuse_conv=nn.Sequential(
            nn.Conv2d(512,embed_dim,kernel_size=1,bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True)
        )
        self.cbam=CBAM(in_channels=512)

    def forward(self, images):
        x=self.stage0_5(images) # [14x14x112]
        f6=self.stage6(x) # [7x7x192]
        f7=self.stage7(f6) # [7x7x320]
        fusion=torch.cat([f6,f7], dim=1) # [7x7x512]
        fusion=self.cbam(fusion)
        if self.refine:
            fusion=self.fuse_conv(fusion)
        B,C,H,W=fusion.shape
        # [B,C,H,W] -> [B,C,N] -> [B,N,C]
        features=fusion.flatten(2) # [B,C,N]
        features=features.permute(0,2,1) # [B,N,C] [Bx49x512]
        return features


