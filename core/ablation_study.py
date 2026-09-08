import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import json

from core.models.autoencoder import EnhancementAutoencoder
from core.train_enhancement import PreprocessedMRIDataset
from core.evaluate_iqa import compute_iqa_metrics

class RicianLoss(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma
        
    def forward(self, pred, target):
        z = (pred * target) / (self.sigma ** 2 + 1e-8)
        
        log_i0 = torch.where(
            z < 1.5,
            (z ** 2) / 4.0,
            z - 0.5 * torch.log(2 * np.pi * z + 1e-8)
        )
        
        loss = (pred ** 2 + target ** 2) / (2 * self.sigma ** 2) - log_i0
        return loss.mean()

def train_ablation(region, loss_type="MSE", epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PreprocessedMRIDataset(region)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = EnhancementAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    if loss_type == "MSE":
        criterion = nn.MSELoss()
    else:
        criterion = RicianLoss(sigma=0.1)
        
    history = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for noisy, clean in loader:
            if torch.sum(clean) == 0: continue
            
            # Convert noisy to Rician noise for proper training evaluation
            # (Adding Rician synthetic noise instead of plain Gaussian)
            n1 = torch.randn_like(clean) * 0.1
            n2 = torch.randn_like(clean) * 0.1
            rician_noisy = torch.sqrt((clean + n1)**2 + n2**2)
            rician_noisy = torch.clamp(rician_noisy, 0.0, 1.0)
            
            rician_noisy = rician_noisy.to(device)
            clean = clean.to(device)
            
            optimizer.zero_grad()
            outputs = model(rician_noisy)
            
            if loss_type == "MSE":
                loss = criterion(outputs, clean)
            else:
                # For Rician loss, we compute the negative log likelihood of the noisy observation
                # given the predicted clean signal.
                loss = criterion(outputs, rician_noisy)
                
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        history.append(epoch_loss / max(batches, 1))
        
    # Evaluate
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for noisy, clean in loader:
            if torch.sum(clean) == 0: continue
            
            n1 = torch.randn_like(clean) * 0.1
            n2 = torch.randn_like(clean) * 0.1
            rician_noisy = torch.sqrt((clean + n1)**2 + n2**2)
            rician_noisy = torch.clamp(rician_noisy, 0.0, 1.0).to(device)
            clean = clean.to(device)
            
            outputs = model(rician_noisy)
            
            img_true = clean[0, 0].cpu().numpy()
            img_pred = outputs[0, 0].cpu().numpy()
            
            m = compute_iqa_metrics(img_true, img_pred)
            all_metrics.append(m)
            break # Just one batch for demo
            
    # Average metrics
    avg_metrics = {}
    for k in all_metrics[0].keys():
        avg_metrics[k] = np.mean([m[k] for m in all_metrics])
        
    return history, avg_metrics, model

def run_experiment():
    os.makedirs("core/outputs/ablation", exist_ok=True)
    
    results = {}
    
    for region in ['brain', 'spine']:
        print(f"Running ablation for {region}...")
        
        # 1. MSE
        print("  Training with MSE...")
        mse_hist, mse_met, mse_model = train_ablation(region, loss_type="MSE", epochs=5)
        
        # 2. Rician
        print("  Training with Rician Loss...")
        ric_hist, ric_met, ric_model = train_ablation(region, loss_type="Rician", epochs=5)
        
        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(mse_hist, label="MSE Loss", color='red')
        
        # Rician loss scale is different (NLL), so we normalize both to [0,1] for visual curve comparison
        mse_norm = (np.array(mse_hist) - min(mse_hist)) / (max(mse_hist) - min(mse_hist) + 1e-8)
        ric_norm = (np.array(ric_hist) - min(ric_hist)) / (max(ric_hist) - min(ric_hist) + 1e-8)
        
        plt.plot(mse_norm, label="MSE Loss (Normalized)", linestyle='--')
        plt.plot(ric_norm, label="Rician Loss (Normalized)", linestyle='--')
        plt.title(f"Convergence Curves: {region.capitalize()}")
        plt.xlabel("Epoch")
        plt.ylabel("Normalized Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"core/outputs/ablation/{region}_loss_curves.png")
        plt.close()
        
        results[region] = {
            "MSE": mse_met,
            "Rician": ric_met
        }
        
    with open("core/outputs/ablation/metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Ablation study complete.")

if __name__ == "__main__":
    run_experiment()
