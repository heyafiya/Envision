import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancementAutoencoder(nn.Module):
    """
    Self-supervised denoising autoencoder / Noise2Void-style CNN.
    Can be used for both Brain and Spine models by instantiating with different weights.
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super(EnhancementAutoencoder, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*2, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*2, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        b = self.bottleneck(p2)
        
        d2 = self.up2(b)
        # Handle padding if shape doesn't match
        diffY = e2.size()[2] - d2.size()[2]
        diffX = e2.size()[3] - d2.size()[3]
        d2 = F.pad(d2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        diffY = e1.size()[2] - d1.size()[2]
        diffX = e1.size()[3] - d1.size()[3]
        d1 = F.pad(d1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.final_conv(d1)
        return torch.sigmoid(out)  # Predicts values in [0, 1]
