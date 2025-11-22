import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import pvlib
from datetime import datetime, timedelta

class SolarDataset(Dataset):
    def __init__(self, images_dir, met_data_path, sequence_length=60, transform=None):
        self.images_dir = images_dir
        self.sequence_length = sequence_length
        self.transform = transform
        
        # 1. Parse Metadata and Data
        self.lat, self.lon, self.alt, self.df = self._parse_met_data(met_data_path)
        
        # 2. Feature Engineering (Physics-Informed)
        self.df = self._add_solar_physics(self.df, self.lat, self.lon, self.alt)
        
        # 3. Filter Night Time (SZA > 85)
        self.df = self.df[self.df['SZA'] <= 85]
        
        # 4. Normalize Features
        # Input Feats: [k_index_lag, Temp, Pressure, SZA, Azimuth, sin_hour, cos_hour]
        
        feature_cols = ['k_index', 'Temp', 'Pressure', 'SZA', 'Azimuth', 'sin_hour', 'cos_hour']
        
        self.feature_cols = feature_cols
        self.mean = self.df[feature_cols].mean()
        self.std = self.df[feature_cols].std()
        
        # Normalize Temp, Pressure, SZA, Azimuth
        cols_to_norm = ['Temp', 'Pressure', 'SZA', 'Azimuth']
        self.df[cols_to_norm] = (self.df[cols_to_norm] - self.mean[cols_to_norm]) / (self.std[cols_to_norm] + 1e-6)
        
        # 5. Match Images
        self.samples = self._match_images()

    def _parse_met_data(self, path):
        # Read header to get coordinates
        with open(path, 'r') as f:
            lines = f.readlines()
            
        lat = 37.0916
        lon = -2.3636
        alt = 490.6 # Updated from prompt
        
        # Try to find them in the header if possible
        for line in lines[:17]:
            if 'Lat' in line:
                try:
                    lat = float(line.split(':')[1].strip())
                except Exception:
                    pass
            if 'Lon' in line:
                try:
                    lon = float(line.split(':')[1].strip())
                except Exception:
                    pass
            if 'Altitude' in line:
                try:
                    alt = float(line.split(':')[1].strip())
                except Exception:
                    pass

        # Read data
        try:
            df = pd.read_csv(path, skiprows=17, sep='\s+', engine='python')
        except Exception:
            df = pd.read_csv(path, skiprows=17, sep='\t')
            
        # Combine Date + Time -> DatetimeIndex
        if 'Date' in df.columns and 'Time' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        else:
            df['Datetime'] = pd.to_datetime(df.iloc[:, 0].astype(str) + ' ' + df.iloc[:, 1].astype(str))
            
        df = df.set_index('Datetime')
        df = df.sort_index()
        
        # Ensure numeric columns
        cols_to_numeric = ['GHI', 'DNI', 'DHI', 'Temp', 'Pressure']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        return lat, lon, alt, df

    def _add_solar_physics(self, df, lat, lon, alt):
        # Calculate Solar Position
        site = pvlib.location.Location(lat, lon, altitude=alt)
        solar_position = site.get_solarposition(df.index)
        
        df['SZA'] = solar_position['zenith']
        df['Azimuth'] = solar_position['azimuth']
        
        # Calculate Clear Sky GHI (Ineichen model)
        clearsky = site.get_clearsky(df.index, model='ineichen')
        df['GHI_cs'] = clearsky['ghi']
        
        # Calculate k* (Clear Sky Index)
        # Avoid division by zero
        df['k_index'] = df['GHI'] / (df['GHI_cs'] + 1e-6)
        # Clamp between 0.0 and 1.2
        df['k_index'] = df['k_index'].clip(0.0, 1.2)
        
        # Time features
        df['hour'] = df.index.hour + df.index.minute / 60.0
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        
        return df

    def _match_images(self):
        samples = []
        if not os.path.exists(self.images_dir):
            return samples
            
        image_files = sorted(os.listdir(self.images_dir))
        
        for img_file in image_files:
            if not img_file.endswith(('.jpg', '.png')):
                continue
                
            try:
                timestamp_str = img_file.split('_')[0]
                img_dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
            except ValueError:
                continue
                
            if img_dt in self.df.index:
                # Check if we have enough history (60 mins)
                start_dt = img_dt - timedelta(minutes=self.sequence_length - 1)
                if start_dt in self.df.index:
                     samples.append((img_file, img_dt))
        
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file, img_dt = self.samples[idx]
        
        # Load Image
        img_path = os.path.join(self.images_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        
        # Apply Circular Mask (Radius 250)
        image = image.resize((512, 512))
        
        # Create mask
        mask = Image.new('L', (512, 512), 0)
        draw = ImageDraw.Draw(mask)
        center = (256, 256)
        radius = 250
        draw.ellipse((center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius), fill=255)
        
        # Apply mask
        image_np = np.array(image)
        mask_np = np.array(mask)
        image_np[mask_np == 0] = 0 # Black out
        image = Image.fromarray(image_np)
        
        if self.transform:
            image = self.transform(image)
        
        # Get Weather Sequence
        end_dt = img_dt
        start_dt = end_dt - timedelta(minutes=self.sequence_length - 1)
        
        weather_slice = self.df.loc[start_dt:end_dt]
        
        # Select features: [k_index_lag, Temp, Pressure, SZA, Azimuth, sin_hour, cos_hour]
        features = weather_slice[self.feature_cols].values
        
        # Handle missing rows
        if len(features) < self.sequence_length:
            pad_len = self.sequence_length - len(features)
            features = np.pad(features, ((pad_len, 0), (0, 0)), mode='edge')
        elif len(features) > self.sequence_length:
             features = features[-self.sequence_length:]
             
        weather_seq = torch.tensor(features, dtype=torch.float32)
        
        # Target k* (at T+0)
        target_k = self.df.loc[img_dt, 'k_index']
        target = torch.tensor([target_k], dtype=torch.float32)
        
        # GHI_cs at T+0 (for reconstruction)
        ghi_cs_t0 = self.df.loc[img_dt, 'GHI_cs']
        ghi_cs = torch.tensor([ghi_cs_t0], dtype=torch.float32)
        
        return image, weather_seq, target, ghi_cs

def get_data_loaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SolarDataset(
        images_dir=config['data']['images_dir'],
        met_data_path=config['data']['met_data_path'],
        sequence_length=config['data']['sequence_length'],
        transform=transform
    )
    
    # Split
    val_size = int(len(dataset) * config['training']['val_split'])
    train_size = len(dataset) - val_size
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    
    return train_loader, val_loader
