#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust Data Processing Pipeline (New Generation)
------------------------------------------------
Author: Assistant
Description:
    This script replaces data_cleaner.py with a time-safe pipeline:
    1. SEGMENTATION: Splits data into episodes based on timestamp gaps (preventing cross-segment labeling).
    2. LABELING: Generates labels based on TIMESTAMPS (not indices), ensuring accuracy even after filtering.
    3. FILTERING: Applies Stationary, Floor-Only, and Quality filters.
    4. AUGMENTATION: Performs synchronized multi-modal augmentation.

Usage:
    python3 scripts/build_dataset.py --input_dir datas --output_dir data_ready --aug_factor 5
"""

import os
import sys
import argparse
import shutil
import random
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# Import helper functions from the original script to maintain feature parity
# Ensure scripts/ is in python path if running from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from data_cleaner import (
        clean_lidar, 
        check_image_quality, 
        is_floor_only_image, 
        augment_image_cv2, 
        ensure_dir, 
        safe_literal_eval
    )
except ImportError:
    print("Error: Could not import from data_cleaner.py. Make sure it exists in the same directory.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Robust Data Pipeline: Segment -> Label -> Filter -> Augment")
    
    # Paths
    parser.add_argument('--input_dir', type=str, default='datas', 
                        help="Path to raw data directory")
    parser.add_argument('--output_dir', type=str, default='data_ready', 
                        help="Output directory for final dataset")
    
    # Segmentation Params
    parser.add_argument('--max_time_gap', type=float, default=1.0, 
                        help="Max time gap (seconds) to consider continuous episode")
    
    # Labeling Params
    parser.add_argument('--collision_dist', type=float, default=0.2, 
                        help="Lidar distance threshold for collision label")
    
    # Cleaning Params
    parser.add_argument('--min_lin_vel', type=float, default=0.03, help="Min linear velocity")
    parser.add_argument('--min_ang_vel', type=float, default=0.03, help="Min angular velocity")
    parser.add_argument('--filter_floor', action='store_true', help="Enable floor-only filtering")
    parser.add_argument('--floor_threshold', type=float, default=0.7, help="Floor detection threshold")
    
    # Augmentation Params
    parser.add_argument('--aug_factor', type=int, default=5, help="Augmentation factor")
    
    return parser.parse_args()

def segment_data(df, max_gap=1.0):
    """
    Split data into episodes based on timestamp continuity.
    Assumes timestamp is in nanoseconds (19 digits) or seconds.
    Auto-detects unit.
    """
    if len(df) == 0:
        return df
        
    # Detect timestamp unit
    ts_0 = df.iloc[0]['timestamp']
    if ts_0 > 1e16: # Nanoseconds
        scale = 1e-9
    elif ts_0 > 1e13: # Microseconds
        scale = 1e-6
    elif ts_0 > 1e10: # Milliseconds
        scale = 1e-3
    else:
        scale = 1.0
        
    # Calculate dt in seconds
    timestamps = df['timestamp'].values.astype(float) * scale
    dt = np.diff(timestamps, prepend=timestamps[0])
    
    # Segment
    episode_ids = []
    curr_ep = 0
    episode_ids.append(curr_ep)
    
    for i in range(1, len(dt)):
        if dt[i] > max_gap:
            curr_ep += 1
        episode_ids.append(curr_ep)
        
    df['episode_id'] = episode_ids
    df['timestamp_sec'] = timestamps
    
    print(f"Data segmented into {curr_ep + 1} episodes.")
    return df

def generate_time_safe_labels(df, args):
    """
    Generate labels based on TIMESTAMP lookups within the same episode.
    Implements conservative labeling strategy - when in doubt, mark as invalid/dangerous.
    """
    print("Generating time-safe labels (conservative strategy)...")
    
    # Time windows for multi-scale
    scales = {
        'short': 0.1,  # 0.1s
        'medium': 0.5, # 0.5s
        'long': 1.0    # 1.0s
    }
    
    # Pre-calculate results containers
    # We will add columns directly to df
    for scale in scales:
        df[f'label_{scale}'] = 0  # Default to safe, but will be overridden when invalid
        df[f'v_{scale}'] = 0.0
        df[f'w_{scale}'] = 0.0
        df[f'valid_{scale}'] = False  # Default to invalid - must prove valid
    
    # Process each episode independently to be safe
    # But for speed, we can iterate full df if sorted
    
    # Convert dataframe columns to numpy for speed
    timestamps = df['timestamp_sec'].values
    episode_ids = df['episode_id'].values
    lidar_series = df['lidar_ranges'].values
    linear_x = df['linear_x'].values
    angular_z = df['angular_z'].values
    
    # We need to parse lidar for collision check. 
    # It's expensive to parse all rows if they might be dropped later,
    # BUT we need full timeline for labels. So we must parse all.
    print("Parsing Lidar for labeling...")
    parsed_lidar = [safe_literal_eval(l) for l in tqdm(lidar_series, desc="Parsing Lidar")]
    
    # Create a lookup helper
    # Since data is sorted by time, we can use searchsorted
    
    n_samples = len(df)
    
    # Results arrays
    labels = {k: np.zeros(n_samples, dtype=int) for k in scales}
    actions_v = {k: np.zeros(n_samples, dtype=float) for k in scales}
    actions_w = {k: np.zeros(n_samples, dtype=float) for k in scales}
    valid_flags = {k: np.zeros(n_samples, dtype=bool) for k in scales}
    
    # Data gap tracking for statistics
    data_gaps = {k: 0 for k in scales}
    invalid_lidar_count = {k: 0 for k in scales}
    
    for scale_name, time_offset in scales.items():
        print(f"Processing scale: {scale_name} (+{time_offset}s)")
        
        target_times = timestamps + time_offset
        
        # Find indices where timestamp >= target_time
        # searchsorted returns the first index where element is >= value
        target_indices = np.searchsorted(timestamps, target_times, side='left')
        
        for i in range(n_samples):
            # Check if index is valid
            idx = target_indices[i]
            
            # If idx is out of bounds, no future data - mark as invalid
            if idx >= n_samples:
                data_gaps[scale_name] += 1
                continue
                
            # Check if found index is actually close to target time (within 0.1s tolerance)
            # and within same episode
            time_diff = abs(timestamps[idx] - target_times[i])
            episode_match = episode_ids[idx] == episode_ids[i]
            
            if (time_diff < 0.1) and episode_match:
                # Found valid target, but still need to check data quality
                
                # 1. Action Target - use nearest valid data
                actions_v[scale_name][i] = linear_x[idx]
                actions_w[scale_name][i] = angular_z[idx]
                
                # 2. Collision Label - Conservative strategy
                # Check ALL frames between i and idx (the window)
                # Collision = ANY frame in window has dist < threshold
                is_dangerous = 0
                valid_window_data = True
                
                for k in range(i, idx + 1):
                    # Robust check for lidar data
                    l_data = parsed_lidar[k]
                    if not l_data or len(l_data) < 10: 
                        invalid_lidar_count[scale_name] += 1
                        valid_window_data = False
                        continue
                    
                    # Filter valid ranges
                    valid_ranges = [r for r in l_data if 0.05 < r < 8.0]
                    if valid_ranges and min(valid_ranges) < args.collision_dist:
                        is_dangerous = 1
                        # Don't break - continue checking for invalid data counting
                
                # Conservative strategy: if any data in window is invalid, mark whole window as dangerous
                if not valid_window_data:
                    is_dangerous = 1  # When in doubt, assume dangerous
                
                labels[scale_name][i] = is_dangerous
                valid_flags[scale_name][i] = True
            else:
                # Data gap or episode boundary - mark as invalid
                data_gaps[scale_name] += 1
                # Keep default values: label=0 (safe but invalid), valid=False
    
    # Print statistics
    print(f"\nLabeling Statistics:")
    for scale in scales:
        valid_count = valid_flags[scale].sum()
        gap_count = data_gaps[scale]
        invalid_lidar = invalid_lidar_count[scale]
        print(f"  {scale:>6}: {valid_count:>6} valid, {gap_count:>6} data gaps, {invalid_lidar:>6} invalid lidar")
    
    # Assign back to DF
    for scale in scales:
        df[f'label_{scale}'] = labels[scale]
        df[f'v_{scale}'] = actions_v[scale]
        df[f'w_{scale}'] = actions_w[scale]
        df[f'valid_{scale}'] = valid_flags[scale]
        
    # Default label is medium
    df['label'] = df['label_medium']
    
    return df

def apply_filters(df, args):
    """
    Apply filters and return a boolean mask (Keep/Drop).
    """
    print("\nApplying filters...")
    keep_mask = np.ones(len(df), dtype=bool)
    
    # 1. Stationary Filter
    is_moving = (df['linear_x'].abs() > args.min_lin_vel) | (df['angular_z'].abs() > args.min_ang_vel)
    keep_mask = keep_mask & is_moving
    print(f"  - Stationary filtered: {len(df) - is_moving.sum()} dropped")
    
    # 2. Label Validity Filter (Crucial!)
    # Conservative strategy: require ALL scales to be valid for robust training
    # This ensures we have consistent data across all time horizons
    has_labels = df['valid_short'] & df['valid_medium'] & df['valid_long']
    keep_mask = keep_mask & has_labels
    print(f"  - Invalid labels filtered: {len(df) - has_labels.sum()} dropped")
    
    # Additional safety check: ensure we have valid collision labels
    # Even if labeled as valid, check for data consistency
    valid_collision_labels = (df['label_short'] >= 0) & (df['label_medium'] >= 0) & (df['label_long'] >= 0)
    keep_mask = keep_mask & valid_collision_labels
    print(f"  - Invalid collision labels filtered: {len(df) - valid_collision_labels.sum()} dropped")
    
    # 3. Image Quality & Floor Filter
    # This is slow, so we iterate only rows that are currently kept
    print("  - Checking image quality & floor detection (this may take a while)...")
    
    # We'll update mask in place
    indices_to_check = df.index[keep_mask].tolist()
    bad_images = []
    floor_images = []
    
    for idx in tqdm(indices_to_check, desc="Image Check"):
        img_path = os.path.join(args.input_dir, df.at[idx, 'image_path'])
        
        # A. Quality Check
        if not check_image_quality(img_path):
            bad_images.append(idx)
            continue
            
        # B. Floor Check (if enabled)
        if args.filter_floor:
            if is_floor_only_image(img_path, args.floor_threshold):
                floor_images.append(idx)
                
    # Update mask
    keep_mask[bad_images] = False
    keep_mask[floor_images] = False
    
    print(f"  - Bad quality images: {len(bad_images)} dropped")
    if args.filter_floor:
        print(f"  - Floor-only images: {len(floor_images)} dropped")
    
    # Final statistics
    total_dropped = len(df) - keep_mask.sum()
    print(f"  - Total samples dropped: {total_dropped} ({total_dropped/len(df)*100:.1f}%)")
    print(f"  - Final samples kept: {keep_mask.sum()}")
        
    return keep_mask

def process_augmentation(df_filtered, args):
    """
    Apply synchronized augmentation and save results.
    """
    print(f"\nAugmenting data (x{args.aug_factor})...")
    
    ensure_dir(args.output_dir)
    aug_img_dir = os.path.join(args.output_dir, 'images')
    ensure_dir(aug_img_dir)
    
    augmented_rows = []
    scales = ['short', 'medium', 'long']
    
    # Iterate over filtered dataframe
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Augmenting"):
        src_img_path = os.path.join(args.input_dir, row['image_path'])
        original_img = cv2.imread(src_img_path)
        if original_img is None: continue
        
        # Parse sensors again (since we stored them as strings/lists in df)
        # Note: df_filtered comes from raw df, so 'lidar_ranges' is likely still string or list depending on parsing state
        # In this script, we did parse it into a list in generate_time_safe_labels BUT we didn't assign it back to df column as object
        # So it's likely still string if we loaded from CSV and didn't overwrite column.
        # Let's safely parse.
        original_lidar = np.array(safe_literal_eval(row['lidar_ranges']))
        original_lidar = clean_lidar(original_lidar) # Apply cleaning here!
        if original_lidar is None: continue
        
        original_imu = np.array(safe_literal_eval(row['imu_data']))
        
        # Create N augmented copies
        for k in range(args.aug_factor):
            aug_row = row.copy()
            aug_img = original_img.copy()
            aug_lidar = original_lidar.copy()
            aug_imu = original_imu.copy()
            
            # --- 1. Synchronized Flip (Horizontal) ---
            if random.random() < 0.5:
                # Image
                aug_img = cv2.flip(aug_img, 1)
                # Lidar (Reverse)
                aug_lidar = np.flip(aug_lidar)
                # IMU/Action (Invert Angular Z)
                aug_row['angular_z'] = -aug_row['angular_z']
                # Invert all multi-scale w targets
                for scale in scales:
                    aug_row[f'w_{scale}'] = -aug_row[f'w_{scale}']
                    
            # --- 2. Sensor Noise ---
            lidar_noise = np.random.normal(0, 0.02, len(aug_lidar))
            aug_lidar = np.clip(aug_lidar + lidar_noise, 0.05, 8.0)
            
            imu_noise = np.random.normal(0, 0.01, len(aug_imu))
            aug_imu = aug_imu + imu_noise
            
            # --- 3. Visual Augmentation ---
            aug_img = augment_image_cv2(aug_img)
            
            # --- Save ---
            # Use original filename + suffix
            orig_name = os.path.basename(row['image_path'])
            aug_name = f"aug_{idx}_{k}_{orig_name}"
            aug_path_abs = os.path.join(aug_img_dir, aug_name)
            cv2.imwrite(aug_path_abs, aug_img)
            
            # Update row
            aug_row['image_path'] = os.path.join('images', aug_name)
            aug_row['lidar_ranges'] = aug_lidar.tolist()
            aug_row['imu_data'] = aug_imu.tolist()
            
            augmented_rows.append(aug_row)
            
    return pd.DataFrame(augmented_rows)

def main():
    args = parse_args()
    print("="*60)
    print("Robust Data Builder Started")
    print("="*60)
    
    input_csv = os.path.join(args.input_dir, 'data.csv')
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return
        
    # 1. Load & Segment
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    df = segment_data(df, args.max_time_gap)
    
    # 2. Label (Time-Safe)
    # This adds label columns to df
    df = generate_time_safe_labels(df, args)
    
    # 3. Filter
    # Get mask of valid rows
    keep_mask = apply_filters(df, args)
    df_filtered = df[keep_mask].copy()
    print(f"\nFiltering complete. {len(df)} -> {len(df_filtered)} samples kept.")
    
    if len(df_filtered) == 0:
        print("No samples left after filtering!")
        return
        
    # 4. Augment & Export
    df_final = process_augmentation(df_filtered, args)
    
    # Save CSV
    out_csv = os.path.join(args.output_dir, 'data.csv')
    df_final.to_csv(out_csv, index=False)
    
    print("\n" + "="*60)
    print(f"Dataset Build Complete!")
    print(f"Output Directory: {args.output_dir}")
    print(f"Final Sample Count: {len(df_final)}")
    print("="*60)

if __name__ == "__main__":
    main()
