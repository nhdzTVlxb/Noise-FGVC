import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_large, ConvNeXt_Large_Weights

class ECA(nn.Module):
    def __init__(self, channels, k_size=5):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): 
        y = self.avg(x)                     
        y = y.squeeze(-1).squeeze(-1)      
        y = y.unsqueeze(1)                 
        y = self.conv(y)                    
        y = self.sigmoid(y).squeeze(1)      
        y = y.unsqueeze(-1).unsqueeze(-1)   
        return x * y


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, learn_p=True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=learn_p)
        self.eps = eps

    def forward(self, x):
        # x: [B,C,H,W]
        p = torch.clamp(self.p, min=1e-1, max=6.0)
        x = torch.clamp(x, min=self.eps).pow(p)
        x = F.avg_pool2d(x, kernel_size=(x.size(-2), x.size(-1))).pow(1.0 / p)
        return x  # [B,C,1,1]


class FineGrainedModel(nn.Module):
    def __init__(self, num_classes =5000, pretrained=True, drop=0.3):
        super().__init__()
        self.backbone = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None)
        self.out_channels = 1536  

        self.eca = ECA(self.out_channels, k_size=5)
        self.gem = GeM(p=3.0, learn_p=True)

        self.head = nn.Sequential(
            nn.Flatten(1),                       
            nn.LayerNorm(self.out_channels),
            nn.Dropout(drop),
            nn.Linear(self.out_channels, num_classes)
        )

    def forward(self, x):
        feat = self.backbone.features(x)    
        feat = self.eca(feat)               
        feat = self.gem(feat)              
        logits = self.head(feat)           
        return logits


def get_model(num_classes =5000, pretrained=True, ckpt_path=None, device="cpu"):
    model = FineGrainedModel(num_classes=num_classes, pretrained=pretrained)
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    else:
        print("使用 ImageNet 预训练初始化（未指定外部权重）。")
    return model
