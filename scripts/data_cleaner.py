#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Processing Pipeline for Collision Model Training
-----------------------------------------------------
Author: Assistant
Description:
    This script performs a 3-stage data processing pipeline:
    1. CLEANING: Filters stationary data, checks sensor integrity, and standardizes formats.
    2. LABELING: Auto-generates collision labels (0/1) based on future Lidar data for ANN.
    3. AUGMENTATION: Expands the dataset using synchronized multi-modal augmentation (Flip, Noise, etc.).

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
    parser.add_argument('--floor_threshold', type=float, default=0.7, help="Threshold for floor detection (0.0-1.0, higher=more strict)")
    parser.add_argument('--floor_debug', action='store_true', help="Enable floor detection debugging output")
    
    # Labeling Params
    parser.add_argument('--future_frames', type=int, default=10, help="Number of future frames to check for collision (approx 0.5s)")
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

def is_floor_only_image(image_path, floor_threshold=0.6, debug=False):
    """
    检测图像是否为纯地板场景 - 基于实际数据调优版本
    基于纹理简单性、颜色一致性和梯度分布来判断
    
    Args:
        image_path: 图像文件路径
        floor_threshold: 地板相似度阈值，0.0-1.0，越高越严格
        debug: 是否输出调试信息
    
    Returns:
        bool: True表示是纯地板，False表示包含其他内容
    """
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            return False  # 无法读取的图像不过滤
        
        h, w = img.shape[:2]
        
        # 转换为HSV色彩空间，更好地处理光照变化
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. 纹理复杂度 (针对小图像优化)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        normalized_texture = laplacian_var / (h * w * 0.01)  # 针对160x120小图像
        
        # 2. 颜色一致性分析
        hue_std = np.std(hsv[:,:,0])
        sat_std = np.std(hsv[:,:,1])
        val_std = np.std(hsv[:,:,2])
        color_consistency_score = (hue_std + sat_std * 0.5 + val_std * 0.3) / 3.0
        
        # 3. 边缘密度 (自适应Canny)
        median_val = np.median(gray)
        lower = int(max(0, 0.4 * median_val))
        upper = int(min(255, 1.2 * median_val))
        edges = cv2.Canny(gray, lower, upper)
        edge_density = np.count_nonzero(edges) / (h * w)
        
        # 4. 梯度方向一致性 (修正计算)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_direction = np.arctan2(sobely, sobelx)
        
        # 只考虑有意义的梯度，避免噪声影响
        meaningful_gradients = gradient_magnitude > np.percentile(gradient_magnitude, 30)
        if np.sum(meaningful_gradients) > 10:
            gradient_direction_std = np.std(gradient_direction[meaningful_gradients])
        else:
            gradient_direction_std = 0
        
        if np.isnan(gradient_direction_std):
            gradient_direction_std = 0
        
        # 5. 频域分析 (改进高频定义)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
        
        # 重新定义高频区域 - 更小的中心区域
        center_h, center_w = h//2, w//2
        high_freq_mask = np.ones_like(magnitude_spectrum)
        # 屏蔽低频区域（中心）
        cv2.circle(high_freq_mask, (center_w, center_h), min(h,w)//6, 0, -1)
        high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask) / np.sum(magnitude_spectrum)
        
        # 基于实际数据调优的阈值和得分计算
        floor_score = 0.0
        
        # 纹理得分 (0.3分) - 主要特征
        if normalized_texture < 5:           # 非常平滑
            floor_score += 0.3
        elif normalized_texture < 15:        # 较平滑
            floor_score += 0.2
        elif normalized_texture < 30:      # 中等纹理
            floor_score += 0.1
        # normalized_texture >= 30: +0.0
            
        # 颜色一致性得分 (0.25分) - 放宽阈值
        if color_consistency_score < 15:     # 非常一致
            floor_score += 0.25
        elif color_consistency_score < 25:   # 较一致
            floor_score += 0.15
        elif color_consistency_score < 35:   # 中等一致
            floor_score += 0.05
        # color_consistency_score >= 35: +0.0
            
        # 边缘密度得分 (0.25分) - 主要特征
        if edge_density < 0.005:             # 极少边缘
            floor_score += 0.25
        elif edge_density < 0.015:           # 较少边缘
            floor_score += 0.15
        elif edge_density < 0.03:            # 中等边缘
            floor_score += 0.05
        # edge_density >= 0.03: +0.0
            
        # 梯度方向一致性得分 (0.1分) - 次要特征
        if gradient_direction_std < 0.6:       # 方向很一致
            floor_score += 0.1
        elif gradient_direction_std < 1.0:     # 方向较一致
            floor_score += 0.05
        # gradient_direction_std >= 1.0: +0.0
            
        # 频域特征得分 (0.1分) - 次要特征，放宽阈值
        if high_freq_energy < 0.15:          # 很少高频
            floor_score += 0.1
        elif high_freq_energy < 0.25:        # 较少高频
            floor_score += 0.05
        # high_freq_energy >= 0.25: +0.0
        
        # 调试输出
        if debug:
            print(f"Image: {os.path.basename(image_path)}")
            print(f"  Texture: {normalized_texture:.2f}, Color: {color_consistency_score:.2f}, Edge: {edge_density:.4f}")
            print(f"  Gradient std: {gradient_direction_std:.2f}, High freq: {high_freq_energy:.3f}")
            print(f"  Floor score: {floor_score:.2f} (threshold: {floor_threshold})")
            print(f"  Result: {'FLOOR' if floor_score >= floor_threshold else 'NOT FLOOR'}")
        
        return floor_score >= floor_threshold
        
    except Exception as e:
        print(f"Warning: Error processing image {image_path}: {e}")
        return False  # 出错时不过滤，保留图像

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
        
    # Prepare output dirs
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
    
    expected_cols = ['lidar_ranges', 'imu_data', 'image_path', 'linear_x', 'angular_z']
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        sys.exit(1)
    
    # 2. Integrity Checks
    print("Performing integrity checks...")
    # Lidar length 360
    valid_lidar = df['lidar_ranges'].apply(lambda x: len(x) == 360)
    # IMU length 9
    valid_imu = df['imu_data'].apply(lambda x: len(x) == 9)
    # Image exists
    def check_image(rel_path):
        abs_path = os.path.join(args.input_dir, rel_path)
        return os.path.exists(abs_path) and os.path.getsize(abs_path) > 0
    
    valid_image = df['image_path'].apply(check_image)
    
    df = df[valid_lidar & valid_imu & valid_image].copy()
    print(f"Dropped {initial_count - len(df)} rows due to integrity checks.")
    
    # 3. Stationary Filter
    print(f"Filtering stationary data (|v| < {args.min_lin_vel} and |w| < {args.min_ang_vel})...")
    is_moving = (df['linear_x'].abs() > args.min_lin_vel) | (df['angular_z'].abs() > args.min_ang_vel)
    df_clean = df[is_moving].copy()
    print(f"Dropped {(~is_moving).sum()} stationary rows.")
    
    # 4. Floor-only Image Filter (Optional)
    if args.filter_floor:
        print(f"Filtering floor-only images (threshold={args.floor_threshold})...")
        
        # 检查每个图像是否为纯地板
        floor_mask = []
        floor_scores = []  # 用于统计
        debug_samples = min(10, len(df_clean))  # 调试样本数
        
        for idx, (_, row) in enumerate(tqdm(df_clean.iterrows(), total=len(df_clean), desc="Checking floor images")):
            img_path = os.path.join(args.input_dir, row['image_path'])
            
            # 前几个样本开启调试模式，帮助用户理解检测效果
            is_debug = args.floor_debug and idx < debug_samples
            is_floor = is_floor_only_image(img_path, args.floor_threshold, debug=is_debug)
            
            floor_mask.append(not is_floor)  # True表示保留，False表示过滤
            floor_scores.append(is_floor)  # 记录检测结果用于统计
        
        # 应用过滤
        df_clean = df_clean[floor_mask].copy()
        dropped_floor = len(floor_mask) - sum(floor_mask)
        
        # 输出统计信息
        if dropped_floor > 0:
            floor_ratio = dropped_floor / len(floor_mask) * 100
            print(f"Dropped {dropped_floor} floor-only images ({floor_ratio:.1f}% of total).")
            
            if args.floor_debug:
                print(f"Floor detection summary:")
                print(f"  Total images checked: {len(floor_mask)}")
                print(f"  Floor images detected: {dropped_floor}")
                print(f"  Non-floor images kept: {sum(floor_mask)}")
                print(f"  Detection threshold: {args.floor_threshold}")
        else:
            print("No floor-only images detected (all images kept).")
            if args.floor_debug:
                print("Tip: If you expected floor images to be filtered, try lowering the threshold with --floor_threshold 0.5")
    
    # 5. Save Images & Data
    print(f"Saving {len(df_clean)} cleaned samples to {args.clean_dir}...")
    if len(df_clean) == 0:
        output_csv = os.path.join(args.clean_dir, 'data.csv')
        df_clean.to_csv(output_csv, index=False)
        return df_clean, output_csv
    
    # Copy images to new directory
    new_image_paths = []
    for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Copying Images"):
        src_path = os.path.join(args.input_dir, row['image_path'])
        img_name = os.path.basename(row['image_path'])
        dst_path = os.path.join(clean_img_dir, img_name)
        
        shutil.copy2(src_path, dst_path)
        new_image_paths.append(os.path.join('images', img_name))
        
    df_clean['image_path'] = new_image_paths
    
    # Save intermediate CSV (without labels yet)
    output_csv = os.path.join(args.clean_dir, 'data.csv')
    df_clean.to_csv(output_csv, index=False)
    
    return df_clean, output_csv

# ==========================================
# Stage 2: Auto Labeling
# ==========================================
def stage_labeling(df, args):
    print("\n" + "="*50)
    print("STAGE 2: Auto-Labeling (ANN Collision Labels)")
    print("="*50)
    
    print(f"Generating labels based on future {args.future_frames} frames (Threshold < {args.collision_dist}m)...")
    
    labels = []
    num_samples = len(df)
    
    # Convert lidar column to list for faster access
    lidar_series = df['lidar_ranges'].tolist()
    
    for i in tqdm(range(num_samples), desc="Labeling"):
        # Look ahead window
        end_idx = min(i + args.future_frames, num_samples)
        future_lidars = lidar_series[i : end_idx]
        
        is_dangerous = 0
        
        for ranges in future_lidars:
            # Filter valid ranges (ignore inf/nan and far clipping)
            valid_ranges = [r for r in ranges if 0.05 < r < 8.0]
            if not valid_ranges:
                continue
            
            frame_min = min(valid_ranges)
            
            if frame_min < args.collision_dist:
                is_dangerous = 1
                break
        
        labels.append(is_dangerous)
        
    df['label'] = labels
    
    # Statistics
    n_danger = sum(labels)
    print(f"Labeling Complete: Safe={num_samples - n_danger}, Dangerous={n_danger} ({n_danger/num_samples*100:.1f}%)")
    
    # Save updated CSV
    output_csv = os.path.join(args.clean_dir, 'data.csv')
    df.to_csv(output_csv, index=False)
    print(f"Saved labeled data to {output_csv}")
    
    return df

# ==========================================
# Stage 3: Data Augmentation
# ==========================================
def augment_image_cv2(img):
    """Random brightness, contrast, blur using OpenCV."""
    # Brightness & Contrast
    if random.random() < 0.5:
        alpha = random.uniform(0.8, 1.2) # Contrast
        beta = random.uniform(-30, 30)   # Brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
    # Blur
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
    
    # Pre-load images is too heavy? No, process row by row.
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
        # Load original image
        src_img_path = os.path.join(args.clean_dir, row['image_path'])
        original_img = cv2.imread(src_img_path)
        
        if original_img is None:
            continue
            
        original_lidar = np.array(row['lidar_ranges'])
        original_imu = np.array(row['imu_data'])
        
        # Create N augmented copies
        for k in range(args.aug_factor):
            aug_row = row.copy()
            aug_img = original_img.copy()
            aug_lidar = original_lidar.copy()
            aug_imu = original_imu.copy()
            aug_w = row['angular_z']
            
            # --- 1. Synchronized Flip (Horizontal) ---
            if random.random() < 0.5:
                # Image Flip
                aug_img = cv2.flip(aug_img, 1)
                # Lidar Flip (Reverse array order)
                # Note: Lidar 0 is usually front/back depending on robot. 
                # Assuming standard 360 lidar where index moves clockwise or ccw.
                # Flipping means reversing the array index order.
                aug_lidar = np.flip(aug_lidar)
                # IMU Angular Z Flip (Invert turn direction)
                aug_w = -aug_w
                aug_row['angular_z'] = aug_w
                
            # --- 2. Lidar Rotation (Roll) ---
            # Random shift 0-360 degrees
            if random.random() < 0.5:
                shift = random.randint(0, 360)
                aug_lidar = np.roll(aug_lidar, shift)
                
            # --- 3. Sensor Noise ---
            # Lidar Noise
            lidar_noise = np.random.normal(0, 0.02, len(aug_lidar))
            aug_lidar = aug_lidar + lidar_noise
            aug_lidar = np.clip(aug_lidar, 0.05, 8.0)
            
            # IMU Noise
            imu_noise = np.random.normal(0, 0.01, len(aug_imu))
            aug_imu = aug_imu + imu_noise
            
            # --- 4. Image Visual Augmentation ---
            aug_img = augment_image_cv2(aug_img)
            
            # --- Save Augmented Sample ---
            # Generate filename: aug_{idx}_{k}_{original_name}
            orig_name = os.path.basename(row['image_path'])
            aug_name = f"aug_{idx}_{k}_{orig_name}"
            aug_path_abs = os.path.join(aug_img_dir, aug_name)
            
            cv2.imwrite(aug_path_abs, aug_img)
            
            # Update Row
            aug_row['image_path'] = os.path.join('images', aug_name)
            aug_row['lidar_ranges'] = aug_lidar.tolist() # Convert back to list for CSV
            aug_row['imu_data'] = aug_imu.tolist()
            
            augmented_rows.append(aug_row)
            
    # Create Augmented DataFrame
    df_aug = pd.DataFrame(augmented_rows)
    
    # Save Augmented CSV
    output_csv = os.path.join(args.aug_dir, 'data.csv')
    df_aug.to_csv(output_csv, index=False)
    
    print(f"\nAugmentation Complete!")
    print(f"Original Samples: {len(df)}")
    print(f"Augmented Samples: {len(df_aug)}")
    print(f"Total Saved to {args.aug_dir}: {len(df_aug)}")

def main():
    args = parse_args()
    
    # Stage 1: Clean
    df_clean, clean_csv_path = stage_cleaning(args)
    
    # Stage 2: Label
    df_labeled = stage_labeling(df_clean, args)
    
    # Stage 3: Augment
    stage_augmentation(df_labeled, args)
    
    print("\n" + "="*50)
    print("ALL STAGES COMPLETED SUCCESSFULLY")
    print(f"1. Cleaned Data: {args.clean_dir}")
    print(f"2. Augmented Data: {args.aug_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
