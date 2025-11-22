import torch
import torch.nn as nn

class LadderFusion(nn.Module):
    def __init__(self, visual_channels, temporal_channels):
        super().__init__()
        self.project = nn.Linear(temporal_channels, visual_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, visual, temporal):
        # visual: (B, C_v, H, W)
        # temporal: (B, C_t) - Assuming this is already pooled to a vector from PyramidTCN
        
        # Project temporal to visual channels
        temp_proj = self.project(temporal) # (B, C_v)
        
        # Reshape for broadcasting: (B, C_v, 1, 1)
        temp_proj = temp_proj.view(temp_proj.shape[0], temp_proj.shape[1], 1, 1)
        
        # Sigmoid Gate: Visual_Out = Visual * Sigmoid(Temp_Proj) + Visual
        gate = self.sigmoid(temp_proj)
        out = visual * gate + visual
        
        return out
