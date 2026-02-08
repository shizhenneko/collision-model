#!/usr/bin/env python3
"""
测试地板检测算法的脚本
用于分析为什么地板图片没有被正确过滤
"""

import cv2
import numpy as np
import os
import sys

# 添加脚本目录到路径
sys.path.append('scripts')

def detailed_floor_analysis(image_path, floor_threshold=0.7):
    """
    详细的地板检测分析，输出所有中间结果
    """
    print(f"\n{'='*60}")
    print(f"分析图片: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print("❌ 无法读取图像")
            return False
        
        h, w = img.shape[:2]
        print(f"图像尺寸: {w}x{h}")
        
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. 纹理复杂度分析
        print(f"\n📊 1. 纹理复杂度分析:")
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        normalized_texture = laplacian_var / (h * w * 0.01)
        print(f"   Laplacian方差: {laplacian_var:.2f}")
        print(f"   归一化纹理: {normalized_texture:.2f}")
        print(f"   阈值判断: normalized_texture < 15 → {normalized_texture < 15}")
        
        # 2. 颜色一致性分析
        print(f"\n🎨 2. 颜色一致性分析:")
        hue_std = np.std(hsv[:,:,0])
        sat_std = np.std(hsv[:,:,1])
        val_std = np.std(hsv[:,:,2])
        color_consistency_score = (hue_std + sat_std * 0.5 + val_std * 0.3) / 3.0
        print(f"   色调标准差: {hue_std:.2f}")
        print(f"   饱和度标准差: {sat_std:.2f}")
        print(f"   亮度标准差: {val_std:.2f}")
        print(f"   综合颜色一致性: {color_consistency_score:.2f}")
        print(f"   阈值判断: color_consistency_score < 20 → {color_consistency_score < 20}")
        
        # 3. 边缘密度分析
        print(f"\n🔍 3. 边缘密度分析:")
        median_val = np.median(gray)
        lower = int(max(0, 0.4 * median_val))
        upper = int(min(255, 1.2 * median_val))
        edges = cv2.Canny(gray, lower, upper)
        edge_density = np.count_nonzero(edges) / (h * w)
        print(f"   Canny阈值: lower={lower}, upper={upper}")
        print(f"   边缘密度: {edge_density:.4f}")
        print(f"   阈值判断: edge_density < 0.015 → {edge_density < 0.015}")
        
        # 4. 梯度方向一致性
        print(f"\n📈 4. 梯度方向一致性:")
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_direction = np.arctan2(sobely, sobelx)
        
        # 计算有意义的梯度方向
        meaningful_gradients = gradient_magnitude > np.mean(gradient_magnitude)
        if np.sum(meaningful_gradients) > 0:
            gradient_direction_std = np.std(gradient_direction[meaningful_gradients])
        else:
            gradient_direction_std = 0
        
        if np.isnan(gradient_direction_std):
            gradient_direction_std = 0
            
        print(f"   有意义梯度像素数: {np.sum(meaningful_gradients)}")
        print(f"   梯度方向标准差: {gradient_direction_std:.2f}")
        print(f"   阈值判断: gradient_direction_std < 0.8 → {gradient_direction_std < 0.8}")
        
        # 5. 频域分析
        print(f"\n🌊 5. 频域分析:")
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
        
        center_h, center_w = h//2, w//2
        high_freq_mask = np.zeros_like(magnitude_spectrum)
        cv2.circle(high_freq_mask, (center_w, center_h), min(h,w)//4, 1, -1)
        high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask) / np.sum(magnitude_spectrum)
        
        print(f"   高频能量比例: {high_freq_energy:.3f}")
        print(f"   阈值判断: high_freq_energy < 0.3 → {high_freq_energy < 0.3}")
        
        # 6. 计算最终得分
        print(f"\n🎯 6. 最终得分计算:")
        floor_score = 0.0
        
        # 纹理得分 (0.25分)
        if normalized_texture < 8:
            floor_score += 0.25
            print(f"   ✅ 纹理得分: +0.25 (normalized_texture < 8)")
        elif normalized_texture < 15:
            floor_score += 0.15
            print(f"   ⚠️  纹理得分: +0.15 (8 ≤ normalized_texture < 15)")
        else:
            print(f"   ❌ 纹理得分: +0.00 (normalized_texture ≥ 15)")
            
        # 颜色一致性得分 (0.25分)
        if color_consistency_score < 12:
            floor_score += 0.25
            print(f"   ✅ 颜色得分: +0.25 (color_consistency_score < 12)")
        elif color_consistency_score < 20:
            floor_score += 0.15
            print(f"   ⚠️  颜色得分: +0.15 (12 ≤ color_consistency_score < 20)")
        else:
            print(f"   ❌ 颜色得分: +0.00 (color_consistency_score ≥ 20)")
            
        # 边缘密度得分 (0.2分)
        if edge_density < 0.008:
            floor_score += 0.2
            print(f"   ✅ 边缘得分: +0.20 (edge_density < 0.008)")
        elif edge_density < 0.015:
            floor_score += 0.1
            print(f"   ⚠️  边缘得分: +0.10 (0.008 ≤ edge_density < 0.015)")
        else:
            print(f"   ❌ 边缘得分: +0.00 (edge_density ≥ 0.015)")
            
        # 梯度方向一致性得分 (0.15分)
        if gradient_direction_std < 0.5:
            floor_score += 0.15
            print(f"   ✅ 梯度得分: +0.15 (gradient_direction_std < 0.5)")
        elif gradient_direction_std < 0.8:
            floor_score += 0.1
            print(f"   ⚠️  梯度得分: +0.10 (0.5 ≤ gradient_direction_std < 0.8)")
        else:
            print(f"   ❌ 梯度得分: +0.00 (gradient_direction_std ≥ 0.8)")
            
        # 频域特征得分 (0.15分)
        if high_freq_energy < 0.2:
            floor_score += 0.15
            print(f"   ✅ 频域得分: +0.15 (high_freq_energy < 0.2)")
        elif high_freq_energy < 0.3:
            floor_score += 0.1
            print(f"   ⚠️  频域得分: +0.10 (0.2 ≤ high_freq_energy < 0.3)")
        else:
            print(f"   ❌ 频域得分: +0.00 (high_freq_energy ≥ 0.3)")
        
        print(f"\n📊 总分: {floor_score:.2f} / 1.00")
        print(f"🎯 阈值: {floor_threshold}")
        is_floor = floor_score >= floor_threshold
        print(f"🏷️  结果: {'🟢 地板' if is_floor else '🔴 非地板'}")
        
        # 保存分析结果图像
        result_dir = "floor_analysis_results"
        os.makedirs(result_dir, exist_ok=True)
        
        # 创建分析结果图
        fig = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        
        # 原图
        fig[0:h, 0:w] = img
        cv2.putText(fig, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # 灰度图
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        fig[0:h, w:w*2] = gray_3ch
        cv2.putText(fig, "Gray", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # 边缘图
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        fig[h:h*2, 0:w] = edges_3ch
        cv2.putText(fig, f"Edges (density: {edge_density:.3f})", (10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # 频谱图
        magnitude_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        magnitude_3ch = cv2.cvtColor(magnitude_norm, cv2.COLOR_GRAY2BGR)
        fig[h:h*2, w:w*2] = magnitude_3ch
        cv2.putText(fig, f"Spectrum (HF: {high_freq_energy:.3f})", (w+10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # 添加结果文字
        result_text = f"Score: {floor_score:.2f} - {'FLOOR' if is_floor else 'NOT FLOOR'}"
        cv2.putText(fig, result_text, (10, h*2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if is_floor else (0,0,255), 2)
        
        result_path = os.path.join(result_dir, f"analysis_{os.path.basename(image_path)}")
        cv2.imwrite(result_path, fig)
        print(f"\n💾 分析结果图已保存: {result_path}")
        
        return is_floor
        
    except Exception as e:
        print(f"❌ 处理图像时出错: {e}")
        return False

if __name__ == "__main__":
    # 测试指定的地板图片
    test_image = "data/cleaned/images/1741601400696880507.jpg"
    
    if os.path.exists(test_image):
        print("开始详细分析地板检测失败原因...")
        result = detailed_floor_analysis(test_image, floor_threshold=0.8)
        print(f"\n检测完成，结果: {'需要过滤' if result else '不需要过滤'}")
    else:
        print(f"❌ 测试图片不存在: {test_image}")
        
        # 检查是否有其他图片可以测试
        if os.path.exists("data/cleaned/images"):
            images = os.listdir("data/cleaned/images")[:5]
            print(f"可用测试图片 (前5个): {images}")
            if images:
                print(f"建议测试: data/cleaned/images/{images[0]}")