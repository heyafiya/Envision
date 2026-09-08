import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
import os

from core.train_enhancement import PreprocessedMRIDataset
from core.models.autoencoder import EnhancementAutoencoder as UNetAutoencoder
from core.evaluate_iqa import compute_iqa_metrics

class PlainAutoencoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super(PlainAutoencoder, self).__init__()
        
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
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*2, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
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
        diffY = e2.size()[2] - d2.size()[2]
        diffX = e2.size()[3] - d2.size()[3]
        d2 = F.pad(d2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # NO SKIP CONNECTION HERE
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        diffY = e1.size()[2] - d1.size()[2]
        diffX = e1.size()[3] - d1.size()[3]
        d1 = F.pad(d1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # NO SKIP CONNECTION HERE
        d1 = self.dec1(d1)
        
        out = self.final_conv(d1)
        return torch.sigmoid(out)

def train_and_eval(model, loader, device, epochs=3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(epochs):
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            if torch.sum(clean) == 0: continue
            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()
            
    # Eval on first batch
    model.eval()
    with torch.no_grad():
        for noisy, clean in loader:
            if torch.sum(clean) == 0: continue
            noisy, clean = noisy.to(device), clean.to(device)
            outputs = model(noisy)
            
            # Take first image in batch
            img_true = clean[0, 0].cpu().numpy()
            img_pred = outputs[0, 0].cpu().numpy()
            
            metrics = compute_iqa_metrics(img_true, img_pred)
            return metrics

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PreprocessedMRIDataset('spine')
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print("Training Plain Autoencoder...")
    plain_model = PlainAutoencoder().to(device)
    plain_metrics = train_and_eval(plain_model, loader, device, epochs=5)
    
    print("Training U-Net Autoencoder...")
    unet_model = UNetAutoencoder().to(device)
    unet_metrics = train_and_eval(unet_model, loader, device, epochs=5)
    
    results = []
    for k in plain_metrics.keys():
        results.append({
            "Metric": k,
            "Plain CNN": plain_metrics[k],
            "U-Net (Skip Connections)": unet_metrics[k]
        })
        
    df = pd.DataFrame(results)
    df.to_csv("core/outputs/unet_comparison.csv", index=False)
    print(df.to_markdown(index=False))
