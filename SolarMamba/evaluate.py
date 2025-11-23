import argparse
import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ladder_net import MambaLadder
from data_loader import get_data_loaders

def evaluate():
    parser = argparse.ArgumentParser(description='Evaluate SolarMamba')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pth')
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    _, val_loader = get_data_loaders(config)
    
    # Load Model
    model = MambaLadder(pretrained=False).to(device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return
        
    model.eval()
    
    # Run Ablation Tests
    print("\nRunning Ablation Tests...")
    results = {}
    
    # 1. Full Model
    print("  Testing Full Model...")
    rmse_full, skill_full = test_model(model, val_loader, device, mode='full')
    results['Full'] = {'rmse': rmse_full, 'skill': skill_full}
    
    # 2. Image Blind
    print("  Testing Image Blind (Time Only)...")
    rmse_img_blind, skill_img_blind = test_model(model, val_loader, device, mode='image_blind')
    results['Image_Blind'] = {'rmse': rmse_img_blind, 'skill': skill_img_blind}
    
    # 3. Time Blind
    print("  Testing Time Blind (Image Only)...")
    rmse_time_blind, skill_time_blind = test_model(model, val_loader, device, mode='time_blind')
    results['Time_Blind'] = {'rmse': rmse_time_blind, 'skill': skill_time_blind}
    
    # Save Results
    import json
    with open('results.json', 'w') as f:
        # Convert numpy types to float for json serialization
        json_results = {}
        for k, v in results.items():
            json_results[k] = {h: float(val) for h, val in v['rmse'].items()} # Save RMSEs
            json_results[k]['skill'] = {h: float(val) for h, val in v['skill'].items()}
            
        json.dump(json_results, f, indent=4)
    print("Saved results.json")
    
    # Plot
    plot_ablation(results)

def test_model(model, dataloader, device, mode='full'):
    # We need to track metrics per horizon
    horizons = [1, 5, 10, 15]
    horizon_indices = {1: 0, 5: 1, 10: 2, 15: 3}
    
    all_preds = {h: [] for h in horizons}
    all_targets = {h: [] for h in horizons}
    all_persistence = {h: [] for h in horizons}
    
    with torch.no_grad():
        for images, weather_seq, targets, ghi_cs in tqdm(dataloader, desc=f"Eval {mode}", leave=False):
            images = images.to(device)
            weather_seq_gpu = weather_seq.to(device)
            
            # Persistence: k*(t) * GHI_cs(t+h)
            # k*(t) is the last value in weather_seq (col 0)
            k_t = weather_seq[:, -1, 0].numpy() # (B,)
            
            # Ablation
            if mode == 'image_blind':
                images = torch.zeros_like(images)
            elif mode == 'time_blind':
                weather_seq_gpu = weather_seq_gpu.clone()
                weather_seq_gpu[:, :, :3] = 0 # Zero out GHI, Temp, Press
            
            # Forward
            preds_k = model(images, weather_seq_gpu) # (B, 4)
            
            # Process per horizon
            for h in horizons:
                idx = horizon_indices[h]
                
                # Reconstruct GHI
                pred_ghi = preds_k[:, idx].cpu().numpy() * ghi_cs[:, idx].numpy()
                actual_ghi = targets[:, idx].cpu().numpy() * ghi_cs[:, idx].numpy()
                persist_ghi = k_t * ghi_cs[:, idx].numpy()
                
                all_preds[h].extend(pred_ghi)
                all_targets[h].extend(actual_ghi)
                all_persistence[h].extend(persist_ghi)
                
    # Calculate Metrics
    rmse_dict = {}
    skill_dict = {}
    
    for h in horizons:
        p = np.array(all_preds[h])
        t = np.array(all_targets[h])
        per = np.array(all_persistence[h])
        
        rmse_model = np.sqrt(np.mean((p - t)**2))
        rmse_per = np.sqrt(np.mean((per - t)**2))
        
        skill = (1 - (rmse_model / rmse_per)) * 100
        
        rmse_dict[h] = rmse_model
        skill_dict[h] = skill
        
    return rmse_dict, skill_dict

def plot_ablation(results):
    plt.figure(figsize=(10, 6))
    horizons = [1, 5, 10, 15]
    
    # Extract skills
    full_skills = [results['Full']['skill'][h] for h in horizons]
    img_blind_skills = [results['Image_Blind']['skill'][h] for h in horizons]
    time_blind_skills = [results['Time_Blind']['skill'][h] for h in horizons]
    
    plt.plot(horizons, full_skills, marker='o', label='Full Model', linewidth=2)
    plt.plot(horizons, img_blind_skills, marker='s', linestyle='--', label='No Image (Time Only)')
    plt.plot(horizons, time_blind_skills, marker='^', linestyle='--', label='No Time (Image Only)')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, label='Persistence')
    
    plt.xlabel('Forecast Horizon (min)')
    plt.ylabel('Skill Score (%)')
    plt.title('Ablation Study: Feature Contribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ablation_chart.png')
    print("Saved ablation_chart.png")

if __name__ == "__main__":
    evaluate()
