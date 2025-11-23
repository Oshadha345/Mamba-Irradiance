import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import pvlib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# --- Configuration ---
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(TEST_DIR, 'metas_09_labels.csv')
IMG_DIR = os.path.join(TEST_DIR, 'metas_09', '08')
OUTPUT_DIR = os.path.join(TEST_DIR, 'output_test')

# Physics Constants
LAT = 37.0916
LON = -2.3636
ALT = 490.6

# Image Processing Constants
IMG_SIZE = (512, 512)
MASK_RADIUS = 250
MASK_CENTER = (256, 256)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_pipeline():
    print("=== Starting Pipeline Component Test ===")
    
    # 1. Load and Parse CSV
    print(f"\n[1] Loading CSV from {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return

    # Read CSV (Simulating _load_all_data logic)
    # The provided CSV seems to have headers based on previous context, let's verify.
    # Assuming standard format: Date,GHI,DNI,DHI,temperature,pressure
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"    Loaded {len(df)} rows.")
        print(f"    Columns: {df.columns.tolist()}")
        
        # Parse Date
        # Check if 'Date' column exists, or if it's the first column
        if 'Date' in df.columns:
            # Try parsing with the expected format YYYYMMDDHHMMSS
            # Based on file list: 20190901080000...
            df['Datetime'] = pd.to_datetime(df['Date'], format='%Y%m%d%H%M%S')
        else:
            print("    'Date' column not found. Trying first column...")
            df['Datetime'] = pd.to_datetime(df.iloc[:, 0], format='%Y%m%d%H%M%S')
            
        df = df.set_index('Datetime').sort_index()
        print("    Timestamp parsing successful.")
        print(f"    Range: {df.index.min()} to {df.index.max()}")
        
    except Exception as e:
        print(f"    Error parsing CSV: {e}")
        return

    # 2. Solar Physics Calculations
    print("\n[2] Testing Solar Physics Calculations (pvlib)...")
    try:
        # Location
        site = pvlib.location.Location(LAT, LON, altitude=ALT)
        
        # Solar Position
        print("    Calculating Solar Position (SZA, Azimuth)...")
        
        # Handle Timezone: Input is UTC+1, pvlib needs UTC
        # Convert index from UTC+1 to UTC
        utc_index = df.index - pd.Timedelta(hours=1)
        utc_index = utc_index.tz_localize('UTC')
        
        solar_position = site.get_solarposition(utc_index)
        
        # Assign back to original index (which is UTC+1)
        df['SZA'] = solar_position['zenith'].values
        df['Azimuth'] = solar_position['azimuth'].values
        
        # Clear Sky GHI
        print("    Calculating Clear Sky GHI (Ineichen)...")
        clearsky = site.get_clearsky(utc_index, model='ineichen')
        df['GHI_cs'] = clearsky['ghi'].values
        
        # k* Calculation
        print("    Calculating Clear Sky Index (k*)...")
        # Avoid division by zero
        df['k_index'] = df['GHI'] / (df['GHI_cs'] + 1e-6)
        df['k_index'] = df['k_index'].clip(0.0, 1.2)
        
        # Time Features
        df['hour'] = df.index.hour + df.index.minute / 60.0
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        
        print("    Physics calculations successful.")
        print(f"    Sample Data (First 3 rows):\n{df[['GHI', 'GHI_cs', 'k_index', 'SZA', 'Azimuth']].head(3)}")
        
    except Exception as e:
        print(f"    Error in physics calculations: {e}")
        return

    # 3. Image Processing & Matching
    print("\n[3] Testing Image Processing & Matching...")
    
    if not os.path.exists(IMG_DIR):
        print(f"Error: Image directory not found at {IMG_DIR}")
        return

    image_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg') or f.endswith('.png')])
    print(f"    Found {len(image_files)} images in {IMG_DIR}")
    
    if not image_files:
        print("    No images found to test.")
        return

    # Test on first 3 images
    for i, img_file in enumerate(image_files[:14]):
        print(f"\n    Processing Image {i+1}: {img_file}")
        
        # Parse timestamp from filename
        # Format: 20190901080000_00160_corrected.jpg
        try:
            ts_str = img_file.split('_')[0]
            img_dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
            print(f"      Parsed Timestamp: {img_dt}")
        except ValueError:
            print(f"      Error parsing timestamp from filename: {img_file}")
            continue
            
        # Check match in CSV
        if img_dt in df.index:
            print("      Match found in CSV!")
            row = df.loc[img_dt]
            print(f"      CSV Data -> GHI: {row['GHI']}, k*: {row['k_index']:.4f}, SZA: {row['SZA']:.2f}")
            
            # Load and Process Image
            img_path = os.path.join(IMG_DIR, img_file)
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    print(f"      Original Size: {img.size}")
                    
                    # Resize
                    img_resized = img.resize(IMG_SIZE)
                    print(f"      Resized to: {img_resized.size}")
                    
                    # Masking
                    mask = Image.new('L', IMG_SIZE, 0)
                    draw = ImageDraw.Draw(mask)
                    # Draw white circle on black background
                    draw.ellipse((MASK_CENTER[0]-MASK_RADIUS, MASK_CENTER[1]-MASK_RADIUS, 
                                  MASK_CENTER[0]+MASK_RADIUS, MASK_CENTER[1]+MASK_RADIUS), fill=255)
                    
                    img_np = np.array(img_resized)
                    mask_np = np.array(mask)
                    
                    # Apply mask
                    img_masked_np = img_np.copy()
                    img_masked_np[mask_np == 0] = 0
                    
                    img_masked = Image.fromarray(img_masked_np)

                    # --- VISUALIZATION: Mark Sun Position ---
                    # Get Solar Position from CSV
                    zenith = row['SZA']
                    azimuth = row['Azimuth']
                    
                    # Projection: Linear Fisheye
                    # Zenith 0 -> Radius 0
                    # Zenith 90 -> Radius 250
                    r_pix = MASK_RADIUS * (zenith / 90.0)
                    
                    # Convert Azimuth (Degrees East of North) to Cartesian
                    # North (0 deg) -> Up (y decreases)
                    # East (90 deg) -> Right (x increases) - Standard Map/Image Convention
                    # x = cx + r * sin(az)
                    # y = cy - r * cos(az)
                    
                    az_rad = np.radians(azimuth)
                    sun_x = MASK_CENTER[0] + r_pix * np.sin(az_rad)
                    sun_y = MASK_CENTER[1] - r_pix * np.cos(az_rad)
                    
                    # Draw Marker
                    draw_viz = ImageDraw.Draw(img_masked)
                    marker_radius = 5
                    # Draw Red Circle for Sun
                    draw_viz.ellipse((sun_x - marker_radius, sun_y - marker_radius, 
                                      sun_x + marker_radius, sun_y + marker_radius), 
                                     fill='red', outline='white')
                    
                    # Add Text
                    # draw_viz.text((sun_x + 10, sun_y), f"Sun: Z{zenith:.1f} A{azimuth:.1f}", fill='yellow')
                    print(f"      Marked Sun Position: Zenith={zenith:.2f}, Azimuth={azimuth:.2f} -> Pixel({sun_x:.1f}, {sun_y:.1f})")
                    # ----------------------------------------
                    
                    # Save for visual verification
                    save_path = os.path.join(OUTPUT_DIR, f"processed_{img_file}")
                    img_masked.save(save_path)
                    print(f"      Saved processed image to: {save_path}")
                    
            except Exception as e:
                print(f"      Error processing image: {e}")
        else:
            print("      No match found in CSV for this timestamp.")

    # 4. Feature Vector Verification
    print("\n[4] Verifying Feature Vector Construction...")
    # Features: ['k_index', 'temperature', 'pressure', 'SZA', 'Azimuth', 'sin_hour', 'cos_hour']
    feature_cols = ['k_index', 'temperature', 'pressure', 'SZA', 'Azimuth', 'sin_hour', 'cos_hour']
    
    # Check if all columns exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"    Error: Missing columns for feature vector: {missing_cols}")
    else:
        print("    All feature columns present.")
        # Normalize (Simulate normalization)
        mean = df[feature_cols].mean()
        std = df[feature_cols].std()
        print("    Calculated Mean and Std for normalization.")
        
        # Show a sample vector
        sample_vec = df.iloc[0][feature_cols].values
        print(f"    Sample Raw Feature Vector (First Row): {sample_vec}")

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_pipeline()
