import torch
import torch.nn as nn
import torchvision.models as models

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

    def forward(self, images):
        x=self.stage0_5(images) # [14x14x112]
        f6=self.stage6(x) # [7x7x192]
        f7=self.stage7(f6) # [7x7x320]
        fusion=torch.cat([f6,f7], dim=1) # [7x7x512]
        if self.refine:
            fusion=self.fuse_conv(fusion)
        B,C,H,W=fusion.shape
        # [B,C,H,W] -> [B,C,N] -> [B,N,C]
        features=fusion.flatten(2) # [B,C,N]
        features=features.permute(0,2,1) # [B,N,C] [Bx49x512]
        return features


