#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Processing Pipeline for Collision Model Training
-----------------------------------------------------
Author: Assistant
Description:
    This script performs a 3-stage data processing pipeline:
    1. CLEANING: Filters stationary data, checks sensor integrity, and standardizes formats.
    2. LABELING: Auto-generates collision labels (0/1) and multi-scale action targets.
    3. AUGMENTATION: Expands the dataset using synchronized multi-modal augmentation.

Usage:
    python3 scripts/data_cleaner.py --input_dir datas --clean_dir data_cleaned --aug_dir data_argument
"""

import os
import sys
import ast
import argparse
import shutil
import random
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Data Processing Pipeline: Clean -> Label -> Augment")
    
    # Paths
    parser.add_argument('--input_dir', type=str, default='datas', 
                        help="Path to raw data directory (containing data.csv and images/)")
    parser.add_argument('--clean_dir', type=str, default='data_cleaned', 
                        help="Output directory for cleaned and labeled data")
    parser.add_argument('--aug_dir', type=str, default='data_argument', 
                        help="Output directory for augmented data")
    
    # Cleaning Params
    parser.add_argument('--min_lin_vel', type=float, default=0.03, help="Min linear velocity to keep")
    parser.add_argument('--min_ang_vel', type=float, default=0.03, help="Min angular velocity to keep")
    parser.add_argument('--filter_floor', action='store_true', help="Filter out floor-only images")
    parser.add_argument('--floor_threshold', type=float, default=0.7, help="Threshold for floor detection")
    parser.add_argument('--floor_debug', action='store_true', help="Enable floor detection debugging output")
    
    # Labeling Params
    parser.add_argument('--future_frames', type=int, default=10, help="Number of future frames for default collision label")
    parser.add_argument('--collision_dist', type=float, default=0.2, help="Lidar distance threshold for collision label (meters)")
    
    # Augmentation Params
    parser.add_argument('--aug_factor', type=int, default=5, help="Number of augmented copies per original sample")
    
    return parser.parse_args()

def safe_literal_eval(val):
    """Safely parse stringified lists from CSV."""
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def clean_lidar(lidar_ranges):
    """
    Lidar data cleaning: length check, clip, spike filter.
    """
    lidar = np.array(lidar_ranges)
    
    # 1. Length check
    if len(lidar) != 360:
        return None
        
    # 2. Clip values (physical constraints)
    lidar = np.clip(lidar, 0.05, 8.0)
    
    # 3. Spike filter (simple smoothing)
    # Check for large jumps between adjacent points which might indicate noise
    # (Except for real edges, but for collision avoidance, smoothing is generally safer)
    # Simple median filter for 1D array
    lidar = np.array(lidar)
    # Roll to check neighbors
    prev_lidar = np.roll(lidar, 1)
    next_lidar = np.roll(lidar, -1)
    
    # If a point is significantly different from BOTH neighbors, replace with average
    spike_mask = (np.abs(lidar - prev_lidar) > 0.5) & (np.abs(lidar - next_lidar) > 0.5)
    lidar[spike_mask] = (prev_lidar[spike_mask] + next_lidar[spike_mask]) / 2.0
    
    return lidar.tolist()

def check_image_quality(image_path):
    """
    Check image quality: brightness (black/overexposed).
    """
    img = cv2.imread(image_path)
    if img is None:
        return False
        
    # Check brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    # Too dark (Black)
    if mean_brightness < 10:
        return False
        
    # Too bright (Overexposed)
    if mean_brightness > 250:
        return False
        
    return True

def is_floor_only_image(image_path, floor_threshold=0.6, debug=False):
    """
    Check if image is floor-only (low texture, consistent color).
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Texture
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        normalized_texture = laplacian_var / (h * w * 0.01)
        
        # 2. Color Consistency
        hue_std = np.std(hsv[:,:,0])
        sat_std = np.std(hsv[:,:,1])
        val_std = np.std(hsv[:,:,2])
        color_consistency_score = (hue_std + sat_std * 0.5 + val_std * 0.3) / 3.0
        
        # 3. Edge Density
        median_val = np.median(gray)
        lower = int(max(0, 0.4 * median_val))
        upper = int(min(255, 1.2 * median_val))
        edges = cv2.Canny(gray, lower, upper)
        edge_density = np.count_nonzero(edges) / (h * w)
        
        floor_score = 0.0
        
        # Simple scoring rules
        if normalized_texture < 5: floor_score += 0.3
        elif normalized_texture < 15: floor_score += 0.2
        elif normalized_texture < 30: floor_score += 0.1
            
        if color_consistency_score < 15: floor_score += 0.25
        elif color_consistency_score < 25: floor_score += 0.15
        elif color_consistency_score < 35: floor_score += 0.05
            
        if edge_density < 0.005: floor_score += 0.25
        elif edge_density < 0.015: floor_score += 0.15
        elif edge_density < 0.03: floor_score += 0.05
        
        # (Simplified other metrics for brevity)
        
        if debug:
            print(f"Img: {os.path.basename(image_path)} | Score: {floor_score:.2f}")
            
        return floor_score >= floor_threshold
        
    except Exception as e:
        print(f"Warning: Error processing image {image_path}: {e}")
        return False

# ==========================================
# Stage 1: Basic Cleaning
# ==========================================
def stage_cleaning(args):
    print("\n" + "="*50)
    print("STAGE 1: Basic Cleaning & Filtering")
    print("="*50)
    
    input_csv = os.path.join(args.input_dir, 'data.csv')
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        sys.exit(1)
        
    ensure_dir(args.clean_dir)
    clean_img_dir = os.path.join(args.clean_dir, 'images')
    ensure_dir(clean_img_dir)
    
    print(f"Loading raw data from {input_csv}...")
    df = pd.read_csv(input_csv)
    initial_count = len(df)
    
    # 1. Parse columns
    print("Parsing sensor data...")
    tqdm.pandas()
    df['lidar_ranges'] = df['lidar_ranges'].progress_apply(safe_literal_eval)
    df['imu_data'] = df['imu_data'].progress_apply(safe_literal_eval)
    
    # 2. Advanced Integrity & Quality Checks
    print("Performing integrity and quality checks...")
    
    # Lidar Cleaning
    def process_lidar(l):
        res = clean_lidar(l)
        return res if res is not None else []
    
    df['lidar_ranges'] = df['lidar_ranges'].progress_apply(process_lidar)
    valid_lidar = df['lidar_ranges'].apply(lambda x: len(x) == 360)
    
    # IMU Cleaning (Length & Clip)
    def process_imu(i):
        if len(i) != 9: return []
        # Clip to reasonable range (e.g., -10 to 10) to avoid crazy outliers
        return np.clip(np.array(i), -10.0, 10.0).tolist()
        
    df['imu_data'] = df['imu_data'].progress_apply(process_imu)
    valid_imu = df['imu_data'].apply(lambda x: len(x) == 9)
    
    # Image Quality Check
    def check_image_wrapper(rel_path):
        abs_path = os.path.join(args.input_dir, rel_path)
        if not os.path.exists(abs_path) or os.path.getsize(abs_path) == 0:
            return False
        return check_image_quality(abs_path)
    
    valid_image = df['image_path'].progress_apply(check_image_wrapper)
    
    # Apply Filters
    df = df[valid_lidar & valid_imu & valid_image].copy()
    print(f"Dropped {initial_count - len(df)} rows due to integrity/quality checks.")
    
    # 3. Stationary Filter
    print(f"Filtering stationary data (|v| < {args.min_lin_vel} and |w| < {args.min_ang_vel})...")
    is_moving = (df['linear_x'].abs() > args.min_lin_vel) | (df['angular_z'].abs() > args.min_ang_vel)
    df_clean = df[is_moving].copy()
    print(f"Dropped {(~is_moving).sum()} stationary rows.")
    
    # 4. Floor-only Image Filter
    if args.filter_floor:
        print(f"Filtering floor-only images (threshold={args.floor_threshold})...")
        floor_mask = []
        for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Checking floor images"):
            img_path = os.path.join(args.input_dir, row['image_path'])
            is_floor = is_floor_only_image(img_path, args.floor_threshold, debug=args.floor_debug)
            floor_mask.append(not is_floor)
        
        df_clean = df_clean[floor_mask].copy()
        dropped_floor = len(floor_mask) - sum(floor_mask)
        print(f"Dropped {dropped_floor} floor-only images.")
    
    # 5. Save
    print(f"Saving {len(df_clean)} cleaned samples to {args.clean_dir}...")
    if len(df_clean) == 0:
        return df_clean, ""
        
    new_image_paths = []
    for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Copying Images"):
        src_path = os.path.join(args.input_dir, row['image_path'])
        img_name = os.path.basename(row['image_path'])
        dst_path = os.path.join(clean_img_dir, img_name)
        shutil.copy2(src_path, dst_path)
        new_image_paths.append(os.path.join('images', img_name))
        
    df_clean['image_path'] = new_image_paths
    output_csv = os.path.join(args.clean_dir, 'data.csv')
    df_clean.to_csv(output_csv, index=False)
    
    return df_clean, output_csv

# ==========================================
# Stage 2: Auto Labeling (Multi-Scale)
# ==========================================
def stage_labeling(df, args):
    print("\n" + "="*50)
    print("STAGE 2: Multi-Scale Labeling")
    print("="*50)
    
    # Multi-scale windows (approx 20fps)
    # Short: 0.1s (~2 frames), Medium: 0.5s (~10 frames), Long: 1.0s (~20 frames)
    scales = {
        'short': 2,
        'medium': 10,
        'long': 20
    }
    
    num_samples = len(df)
    lidar_series = df['lidar_ranges'].tolist()
    
    # Prepare columns
    labels = {k: [] for k in scales.keys()} # Collision labels
    actions = {k: [] for k in scales.keys()} # Action targets [v, w]
    
    print("Generating multi-scale labels...")
    
    for i in tqdm(range(num_samples), desc="Labeling"):
        # 1. Collision Labels (for ANN)
        # Check future lidar for collision in different windows
        for scale_name, window_size in scales.items():
            end_idx = min(i + window_size, num_samples)
            future_lidars = lidar_series[i : end_idx]
            
            is_dangerous = 0
            for ranges in future_lidars:
                valid_ranges = [r for r in ranges if 0.05 < r < 8.0]
                if valid_ranges and min(valid_ranges) < args.collision_dist:
                    is_dangerous = 1
                    break
            labels[scale_name].append(is_dangerous)
            
        # 2. Action Targets (for VOA Multi-Scale)
        # Get action at future time point
        for scale_name, window_size in scales.items():
            target_idx = min(i + window_size, num_samples - 1)
            target_v = df.iloc[target_idx]['linear_x']
            target_w = df.iloc[target_idx]['angular_z']
            actions[scale_name].append([target_v, target_w])
            
    # Add to DataFrame
    # Default label (medium)
    df['label'] = labels['medium']
    
    # Multi-scale columns
    for scale in scales:
        df[f'label_{scale}'] = labels[scale]
        # Unpack action lists to columns
        action_list = actions[scale]
        df[f'v_{scale}'] = [a[0] for a in action_list]
        df[f'w_{scale}'] = [a[1] for a in action_list]
        
    print(f"Labels generated. Medium scale danger rate: {sum(labels['medium'])/num_samples*100:.1f}%")
    
    output_csv = os.path.join(args.clean_dir, 'data.csv')
    df.to_csv(output_csv, index=False)
    
    return df

# ==========================================
# Stage 3: Data Augmentation
# ==========================================
def augment_image_cv2(img):
    """Random brightness, contrast, blur."""
    if random.random() < 0.5:
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-30, 30)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img

def stage_augmentation(df, args):
    print("\n" + "="*50)
    print(f"STAGE 3: Data Augmentation (x{args.aug_factor})")
    print("="*50)
    
    ensure_dir(args.aug_dir)
    aug_img_dir = os.path.join(args.aug_dir, 'images')
    ensure_dir(aug_img_dir)
    
    augmented_rows = []
    
    scales = ['short', 'medium', 'long']
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
        src_img_path = os.path.join(args.clean_dir, row['image_path'])
        original_img = cv2.imread(src_img_path)
        if original_img is None: continue
            
        original_lidar = np.array(row['lidar_ranges'])
        original_imu = np.array(row['imu_data'])
        
        for k in range(args.aug_factor):
            aug_row = row.copy()
            aug_img = original_img.copy()
            aug_lidar = original_lidar.copy()
            aug_imu = original_imu.copy()
            
            # --- 1. Synchronized Flip (Horizontal) ---
            if random.random() < 0.5:
                # Image: Flip Horizontal
                aug_img = cv2.flip(aug_img, 1)
                
                # Lidar: Flip (Reverse array)
                aug_lidar = np.flip(aug_lidar)
                
                # Actions: Invert Angular Z (w) for ALL scales
                aug_row['angular_z'] = -aug_row['angular_z'] # Current
                for scale in scales:
                    aug_row[f'w_{scale}'] = -aug_row[f'w_{scale}']
                    
                # IMU: You might want to flip IMU angular z too if it exists in the array
                # Assuming index 5 is w_z in typical 9-dof [ax,ay,az, wx,wy,wz, mx,my,mz]
                # But safer to just add noise as structure varies.
                
            # --- 2. Lidar Shift (Removed large rotation) ---
            # Large random rotation breaks image-lidar correspondence.
            # Only small jitter allowed if needed, but skipped here for safety.
            
            # --- 3. Sensor Noise ---
            lidar_noise = np.random.normal(0, 0.02, len(aug_lidar))
            aug_lidar = np.clip(aug_lidar + lidar_noise, 0.05, 8.0)
            
            imu_noise = np.random.normal(0, 0.01, len(aug_imu))
            aug_imu = aug_imu + imu_noise
            
            # --- 4. Image Visual Augmentation ---
            aug_img = augment_image_cv2(aug_img)
            
            # --- Save ---
            orig_name = os.path.basename(row['image_path'])
            aug_name = f"aug_{idx}_{k}_{orig_name}"
            aug_path_abs = os.path.join(aug_img_dir, aug_name)
            cv2.imwrite(aug_path_abs, aug_img)
            
            aug_row['image_path'] = os.path.join('images', aug_name)
            aug_row['lidar_ranges'] = aug_lidar.tolist()
            aug_row['imu_data'] = aug_imu.tolist()
            
            augmented_rows.append(aug_row)
            
    df_aug = pd.DataFrame(augmented_rows)
    output_csv = os.path.join(args.aug_dir, 'data.csv')
    df_aug.to_csv(output_csv, index=False)
    
    print(f"\nAugmentation Complete!")
    print(f"Original: {len(df)} -> Augmented: {len(df_aug)}")

def main():
    args = parse_args()
    df_clean, _ = stage_cleaning(args)
    if len(df_clean) > 0:
        df_labeled = stage_labeling(df_clean, args)
        stage_augmentation(df_labeled, args)
    else:
        print("No data left after cleaning.")

    print("\n" + "="*50)
    print("ALL STAGES COMPLETED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    main()
