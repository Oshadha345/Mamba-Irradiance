import torch
import torch.nn as nn
from mambavision.models.mamba_vision import mamba_vision_B
from .temporal import PyramidTCN
from .fusion import LadderFusion

class MambaLadder(nn.Module):
    def __init__(self, pretrained=True, model_path=None):
        super().__init__()
        
        # 1. Visual Backbone (MambaVision-B)
        # We need to load it such that we can access intermediate features.
        # MambaVision-B dims: [128, 256, 512, 1024] for stages 1, 2, 3, 4
        if model_path:
            self.visual_backbone = mamba_vision_B(pretrained=pretrained, model_path=model_path)
        else:
            self.visual_backbone = mamba_vision_B(pretrained=pretrained)
        
        # Hooks to capture features
        self.features = {}
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook

        # Register hooks on the levels (stages)
        # MambaVision has self.levels which is a ModuleList
        for i, level in enumerate(self.visual_backbone.levels):
            level.register_forward_hook(get_activation(f'stage_{i+1}'))
            
        # 2. Temporal Backbone
        self.temporal_backbone = PyramidTCN(input_channels=7, embedding_dim=128)
        
        # 3. Fusion Blocks
        # MambaVision-B dims: 128, 256, 512, 1024
        self.fusion1 = LadderFusion(visual_channels=128, temporal_channels=128)
        self.fusion2 = LadderFusion(visual_channels=256, temporal_channels=128)
        self.fusion3 = LadderFusion(visual_channels=512, temporal_channels=128)
        self.fusion4 = LadderFusion(visual_channels=1024, temporal_channels=128)
        
        # 4. Head
        # GlobalAvgPool -> Concat -> MLP -> Output
        # We pool each fusion output.
        # Concat size = 128 + 256 + 512 + 1024 = 1920
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(1920, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1), # Output k*
            nn.Sigmoid() # k* is between 0 and 1.2 (scaled later)
        )

    def forward(self, image, weather_seq):
        # Clear previous features
        self.features = {}
        
        # Visual Forward
        _ = self.visual_backbone.forward_features(image)
        
        # Retrieve features
        f1 = self.features['stage_1'] # (B, 128, H/4, W/4)
        f2 = self.features['stage_2'] # (B, 256, H/8, W/8)
        f3 = self.features['stage_3'] # (B, 512, H/16, W/16)
        f4 = self.features['stage_4'] # (B, 1024, H/32, W/32)
        
        # Temporal Forward
        t_feats = self.temporal_backbone(weather_seq) # [t1, t2, t3, t4]
        
        # Fusion
        out1 = self.fusion1(f1, t_feats[0])
        out2 = self.fusion2(f2, t_feats[1])
        out3 = self.fusion3(f3, t_feats[2])
        out4 = self.fusion4(f4, t_feats[3])
        
        # Pooling and Concat
        p1 = self.avg_pool(out1).flatten(1)
        p2 = self.avg_pool(out2).flatten(1)
        p3 = self.avg_pool(out3).flatten(1)
        p4 = self.avg_pool(out4).flatten(1)
        
        concat = torch.cat([p1, p2, p3, p4], dim=1)
        
        # Prediction
        # Scale sigmoid output by 1.2 to allow k* > 1
        pred = self.head(concat) * 1.2
        
        return pred
