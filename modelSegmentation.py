import torch
from torch import nn
from torchvision import models

class SegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(SegmentationModel, self).__init__()
        
        # Load a pre-trained backbone (e.g., ResNet)
        self.backbone = models.resnet18(pretrained=True)
        
        # Modify the last layer to output desired number of channels for segmentation
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Conv2d(in_features, num_classes, kernel_size=1)
        
        # Define upsampling layer to resize the output to the input size
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Forward pass through the backbone
        x = self.backbone(x)
        
        # Upsample the output to the input size
        x = self.upsample(x)
        
        return x
