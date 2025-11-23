import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import argparse

# Add parent directory to path to allow importing mambavision
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ladder_net import MambaLadder
from data_loader import get_data_loaders

def train():
    parser = argparse.ArgumentParser(description='Train SolarMamba')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save checkpoints')
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
        
    # Check Data Existence (Basic check on root)
    if config['env'] == 'server':
        root = config['data']['server_root']
    else:
        root = config['data']['local_root']
        
    if not os.path.exists(root):
        # If mock data generation is needed, we might want to warn
        print(f"Warning: Root directory {root} does not exist. Ensure data is present or run mock_data.py.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data Loaders
    train_loader, val_loader = get_data_loaders(config)
    
    # Model
    model_path = config['model'].get('pretrained_weights', None)
    model = MambaLadder(pretrained=True, model_path=model_path).to(device)
    
    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']), weight_decay=float(config['training']['weight_decay']))
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training Loop
    epochs = config['training']['epochs']
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, weather_seq, targets, _ in pbar:
            images = images.to(device)
            weather_seq = weather_seq.to(device)
            targets = targets.to(device) # Shape: (B, 4)
            
            optimizer.zero_grad()
            outputs = model(images, weather_seq) # Shape: (B, 4)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds_k = []
        all_targets_k = []
        all_ghi_cs = []
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, weather_seq, targets, ghi_cs in pbar_val:
                images = images.to(device)
                weather_seq = weather_seq.to(device)
                targets = targets.to(device)
                
                outputs = model(images, weather_seq)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                
                all_preds_k.append(outputs.cpu().numpy())
                all_targets_k.append(targets.cpu().numpy())
                all_ghi_cs.append(ghi_cs.numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Metrics on Reconstructed GHI (Average across all horizons for summary)
        all_preds_k = np.concatenate(all_preds_k, axis=0) # (N, 4)
        all_targets_k = np.concatenate(all_targets_k, axis=0) # (N, 4)
        all_ghi_cs = np.concatenate(all_ghi_cs, axis=0) # (N, 4)
        
        # Reconstruct GHI
        pred_ghi = all_preds_k * all_ghi_cs
        actual_ghi = all_targets_k * all_ghi_cs
        
        rmse = np.sqrt(np.mean((pred_ghi - actual_ghi)**2))
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val RMSE (Avg): {rmse:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model to {save_path}")

    # Visualize one batch after training
    # print("Generating visualization...")
    # visualize_batch(model, val_loader) # Visualization needs update for multi-horizon, skipping for now or update later

def visualize_batch(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    images, weather_seq, targets, ghi_cs = next(iter(loader))
    images = images.to(device)
    weather_seq = weather_seq.to(device)
    
    with torch.no_grad():
        preds_k = model(images, weather_seq)
        
    # Visualize first sample in batch
    img = images[0].cpu().permute(1, 2, 0).numpy()
    # Un-normalize image for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    # k* trend
    k_trend = weather_seq[0, :, 0].cpu().numpy() # k_index is col 0
    pred_k_val = preds_k[0].item()
    actual_k_val = targets[0].item()
    
    ghi_cs_val = ghi_cs[0].item()
    pred_ghi = pred_k_val * ghi_cs_val
    actual_ghi = actual_k_val * ghi_cs_val
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("ASI Image (Masked)")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.plot(k_trend, label='Past 60m k*')
    plt.scatter(60, pred_k_val, color='red', label='Predicted k*', marker='x', s=100)
    plt.scatter(60, actual_k_val, color='green', label='Actual k*', marker='o')
    plt.title(f"Forecast\nPred GHI: {pred_ghi:.1f}, Actual: {actual_ghi:.1f} W/m2")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.3)
    
    plt.tight_layout()
    plt.savefig("visualization.png")
    print("Saved visualization.png")

if __name__ == "__main__":
    train()
