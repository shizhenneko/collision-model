import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from dataset import CollisionDataset, split_dataset_train_val_test
from models import VOAModel, CollisionModel
import time

def main():
    # ================= 配置参数 (Configuration) =================
    data_roots = [
        '../data_ready',
        '../data_ready_1'
    ]
    
    batch_size = 32
    num_epochs = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # RWR (Reward-Weighted Regression) 配置 - 强制启用
    # 必须先训练 train_ann.py 并生成 ann_best.pth
    ANN_CHECKPOINT = '../checkpoints/ann_best.pth'
    RWR_TEMPERATURE = 0.5 # 推荐值 0.5，平衡安全性与样本多样性
    
    # 多尺度损失权重 [short, medium, long]
    # 远期预测不确定性大，权重降低
    MULTI_SCALE_WEIGHTS = [1.0, 0.8, 0.5] 
    
    # 策略头与恢复头的 Loss 比例
    LAMBDA_RECOVERY = 1.0
    
    # ================= 数据准备 (Data Preparation) =================
    datasets = []
    for root in data_roots:
        full_path = os.path.abspath(root)
        if os.path.exists(full_path):
            print(f"Found dataset at: {full_path}")
            try:
                ds = CollisionDataset(full_path)
                datasets.append(ds)
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
                
    if not datasets:
        print("No valid datasets found. Exiting.")
        return

    full_dataset = ConcatDataset(datasets)
    # 使用重构后的划分函数 (70/15/15)
    train_set, val_set, test_set = split_dataset_train_val_test(full_dataset, val_ratio=0.15, test_ratio=0.15)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # test_loader 可以留给评估脚本使用
    
    # ================= 模型构建 (Model Building) =================
    # 双头 VOA 模型
    model = VOAModel().to(device)
    
    # 加载 ANN 模型作为 Reward Function (强制)
    if not os.path.exists(ANN_CHECKPOINT):
        raise FileNotFoundError(f"ANN Checkpoint not found at {ANN_CHECKPOINT}. Please run train_ann.py first.")
        
    print(f"Loading ANN model for RWR from {ANN_CHECKPOINT}")
    ann_model = CollisionModel().to(device)
    ann_model.load_state_dict(torch.load(ANN_CHECKPOINT))
    ann_model.eval()
    for param in ann_model.parameters():
        param.requires_grad = False
            
    # ================= 优化器与损失函数 (Optimizer & Loss) =================
    criterion = nn.MSELoss(reduction='none')
    
    # 分层学习率 (Differential Learning Rate)
    # Backbone (Vision): 1e-5 (微调)
    # Others (Lidar, IMU, Heads): 1e-4 (正常训练)
    optimizer = optim.Adam([
        {'params': model.vis_encoder.parameters(), 'lr': 1e-5},
        {'params': model.lidar_encoder.parameters(), 'lr': 1e-4},
        {'params': model.imu_encoder.parameters(), 'lr': 1e-4},
        {'params': model.fusion.parameters(), 'lr': 1e-4},
        {'params': model.policy_head.parameters(), 'lr': 1e-4},
        {'params': model.recovery_head.parameters(), 'lr': 1e-4}
    ])
    
    # ================= 训练循环 (Training Loop) =================
    best_val_loss = float('inf')
    save_dir = '../checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    print("Start Training Dual-Head VOA with RWR...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        # 确保 VisionEncoder 的 BN 层行为正确 (虽然参数冻结了，但如果开放了深层，BN 仍需更新统计量)
        # 如果是完全冻结，应该 eval()；这里我们部分冻结，部分微调，所以保持 train()
        
        train_loss = 0.0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            lidars = batch['lidar'].to(device)
            imus = batch['imu'].to(device)
            
            # 目标
            target_policy = batch['action'].to(device) # [B, 2]
            target_recovery = batch['multi_scale_action'].to(device) # [B, 6]
            
            # 有效性掩码
            raw_mask = batch['valid_mask'].to(device) # [B, 3] for recovery
            # Policy Head 始终假设有效 (除非是 Episode 极末尾，但一般都有效)
            # Recovery Head Mask: [v_s, w_s, v_m, w_m, v_l, w_l]
            mask_recovery = torch.repeat_interleave(raw_mask, 2, dim=1) # [B, 6]
            
            optimizer.zero_grad()
            
            # 1. Forward Pass (Dual Head)
            outputs = model(images, lidars, imus)
            pred_policy = outputs['policy'] # [B, 2]
            pred_recovery = outputs['recovery'] # [B, 6]
            
            # 2. RWR Weight Calculation
            with torch.no_grad():
                risk_scores = ann_model(images, lidars, imus) # [B, 1]
                # Reward: 越安全(risk越低)奖励越高
                rewards = 1.0 - risk_scores
                # Weight: exp(R / T)
                rwr_weights = torch.exp(rewards / RWR_TEMPERATURE) # [B, 1]
            
            # 3. Policy Loss (Single Scale)
            loss_policy = criterion(pred_policy, target_policy) # [B, 2]
            loss_policy = (loss_policy.mean(dim=1, keepdim=True) * rwr_weights).mean()
            
            # 4. Recovery Loss (Multi Scale)
            # 将 6维 拆分为 3组 [B, 2]
            loss_recovery_raw = criterion(pred_recovery, target_recovery) # [B, 6]
            # 应用 Valid Mask
            loss_recovery_masked = loss_recovery_raw * mask_recovery
            
            # 应用时间尺度权重 (Alpha)
            # [v_s, w_s] * w1, [v_m, w_m] * w2, [v_l, w_l] * w3
            # 构建权重向量 [B, 6]
            scale_weights = torch.tensor(
                [MULTI_SCALE_WEIGHTS[0]]*2 + [MULTI_SCALE_WEIGHTS[1]]*2 + [MULTI_SCALE_WEIGHTS[2]]*2,
                device=device
            ).view(1, 6)
            
            loss_recovery_weighted = loss_recovery_masked * scale_weights
            
            # 对每个样本求和，再应用 RWR
            # [B, 6] -> [B, 1]
            loss_recovery_per_sample = loss_recovery_weighted.sum(dim=1, keepdim=True)
            # 归一化：除以有效的 mask 数量 (避免全0导致除0)
            valid_counts = mask_recovery.sum(dim=1, keepdim=True) + 1e-6
            loss_recovery_per_sample = loss_recovery_per_sample / valid_counts
            
            loss_recovery = (loss_recovery_per_sample * rwr_weights).mean()
            
            # 5. Total Loss
            total_loss = loss_policy + LAMBDA_RECOVERY * loss_recovery
            
            # 6. Backward
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item() * images.size(0)
            
        train_loss /= len(train_set)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                lidars = batch['lidar'].to(device)
                imus = batch['imu'].to(device)
                target_policy = batch['action'].to(device)
                target_recovery = batch['multi_scale_action'].to(device)
                raw_mask = batch['valid_mask'].to(device)
                
                outputs = model(images, lidars, imus)
                pred_policy = outputs['policy']
                pred_recovery = outputs['recovery']
                
                # Validation 不一定非要加 RWR，但为了指标一致性，通常加上
                # 或者只看原始 MSE。这里为了简单，看原始 MSE
                
                loss_policy = criterion(pred_policy, target_policy).mean()
                
                mask_recovery = torch.repeat_interleave(raw_mask, 2, dim=1)
                loss_recovery = (criterion(pred_recovery, target_recovery) * mask_recovery).sum() / (mask_recovery.sum() + 1e-6)
                
                total_loss = loss_policy + loss_recovery # 简单求和作为指标
                
                val_loss += total_loss.item() * images.size(0)
                
        val_loss /= len(val_set)
        
        end_time = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Time: {end_time - start_time:.1f}s "
              f"Train Loss (RWR): {train_loss:.4f} | "
              f"Val Loss (MSE): {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, 'voa_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  -> Model saved to {save_path}")

    print("Training Completed.")

if __name__ == '__main__':
    main()
