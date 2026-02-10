import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from dataset import CollisionDataset, split_dataset_train_val_test
from models import CollisionModel, VOAModel
from train_ann import evaluate_binary

def evaluate_voa(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_recovery_loss = 0.0
    total_count = 0
    
    # 记录反归一化后的误差 (假设归一化参数 v_max=0.3, w_max=1.5)
    # 这里我们简单地在归一化空间计算 MSE，也可以还原
    action_scale = torch.tensor([0.3, 1.5], device=device)
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            lidars = batch['lidar'].to(device)
            imus = batch['imu'].to(device)
            
            # VOA 目标
            # policy target: 当前时刻动作 [B, 2]
            target_policy = batch['action'].to(device)
            # recovery target: 多尺度动作 [B, 6]
            target_recovery = batch['multi_scale_action'].to(device)
            # mask
            valid_mask = batch['valid_mask'].to(device) # [B, 3]
            
            # Model Forward
            outputs = model(images, lidars, imus)
            pred_policy = outputs['policy']     # [B, 2]
            pred_recovery = outputs['recovery'] # [B, 6]
            
            # 1. Policy Loss (MSE)
            loss_policy = criterion(pred_policy, target_policy).mean()
            
            # 2. Recovery Loss (Masked MSE)
            # pred_recovery: [B, 6] -> [B, 3, 2]
            pred_rec_reshaped = pred_recovery.view(-1, 3, 2)
            target_rec_reshaped = target_recovery.view(-1, 3, 2)
            
            # 计算每个时间步的 MSE [B, 3]
            # dim=2 (v, w)
            loss_per_step = ((pred_rec_reshaped - target_rec_reshaped) ** 2).mean(dim=2)
            
            # 应用掩码: valid_mask [B, 3]
            # 避免除以0
            masked_loss = (loss_per_step * valid_mask).sum() / (valid_mask.sum() + 1e-6)
            
            loss = loss_policy + masked_loss
            
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_policy_loss += loss_policy.item() * batch_size
            total_recovery_loss += masked_loss.item() * batch_size
            total_count += batch_size

    return {
        'loss': total_loss / max(total_count, 1),
        'policy_mse': total_policy_loss / max(total_count, 1),
        'recovery_mse': total_recovery_loss / max(total_count, 1)
    }

def main():
    # ================= 配置参数 =================
    # 数据集路径 - 自动扫描上级目录中符合命名规则的数据集目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 自动判断模型类型和路径
    # 优先检查命令行参数，或者默认路径
    # 这里简单起见，我们检测是否存在 voa_best.pth 或 ann_best.pth
    # 并根据用户当前的上下文，假设用户想评估 voa_best.pth (因为它在 Read 输出中出现了)
    
    # 默认寻找 voa
    default_model = 'voa_best.pth'
    # 如果 voa 不存在，找 ann
    if not os.path.exists(os.path.join(base_dir, 'checkpoints', default_model)):
        default_model = 'ann_best.pth'
        
    model_name = default_model
    model_path = os.path.join(base_dir, 'checkpoints', model_name)
    
    is_voa = 'voa' in model_name.lower()
    print(f"Target Model: {model_name} (Mode: {'VOA' if is_voa else 'ANN'})")
    
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ================= 数据准备 =================
    # 警告：为了保证测试集与训练时一致，这里必须使用与训练时完全相同的数据集列表和顺序
    # 默认只使用 data_ready 和 data_ready_1
    datasets = []
    
    # 确保顺序固定
    sorted_roots = []
    if os.path.exists(os.path.join(base_dir, 'data_ready')):
        sorted_roots.append(os.path.join(base_dir, 'data_ready'))
    if os.path.exists(os.path.join(base_dir, 'data_ready_1')):
        sorted_roots.append(os.path.join(base_dir, 'data_ready_1'))
        
    # 如果需要扫描其他目录，请确保顺序与训练时一致
    # 这里我们暂时注释掉自动扫描，以免破坏划分一致性
    # for item in sorted(os.listdir(base_dir)):
    #     if item.startswith('data_ready') and os.path.isdir(os.path.join(base_dir, item)):
    #         path = os.path.join(base_dir, item)
    #         if path not in sorted_roots:
    #             sorted_roots.append(path)

    print(f"Using datasets: {sorted_roots}")

    for root in sorted_roots:
        try:
            # 如果是 VOA 模式，需要开启 normalize_action
            # VOA 训练时使用了 ACTION_NORM = True
            ds = CollisionDataset(
                root, 
                normalize_action=True if is_voa else False,
                v_max=0.3, # VOA 默认值
                w_max=1.5  # VOA 默认值
            )
            datasets.append(ds)
        except Exception as e:
            print(f"Error loading {root}: {e}")
    
    if not datasets:
        print("No datasets found. Exiting.")
        return

    full_dataset = ConcatDataset(datasets)
    print(f"Total samples: {len(full_dataset)}")

    # 分割数据集
    # 注意：train_ratio 不是参数，训练集是剩余部分
    # 必须保持 val_ratio 和 test_ratio 与训练时一致 (0.15, 0.15)
    train_dataset, val_dataset, test_dataset = split_dataset_train_val_test(
        full_dataset, 
        val_ratio=0.15, 
        test_ratio=0.15
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ================= 模型加载 =================
    if is_voa:
        # VOA 模型
        model = VOAModel(
            action_output_mode='tanh_norm', # 假设训练时用了 tanh_norm，train_voa.py 默认如此
            action_v_max=0.3,
            action_w_max=1.5
        ).to(device)
    else:
        # ANN 模型
        model = CollisionModel().to(device)
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # 兼容保存的是 state_dict 还是整个 checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print(f"Model file not found at {model_path}")
        return

    # ================= 评估 =================
    
    print("\nStarting evaluation on Test Set...")
    
    if is_voa:
        # VOA 评估
        criterion = nn.MSELoss(reduction='none') # 与训练时一致
        metrics = evaluate_voa(model, test_loader, device, criterion)
        
        print("\n================ Evaluation Results (VOA) ================")
        print(f"Total Loss: {metrics['loss']:.4f}")
        print(f"Policy MSE: {metrics['policy_mse']:.4f}")
        print(f"Recovery MSE: {metrics['recovery_mse']:.4f}")
        print("==========================================================")
        
    else:
        # ANN 评估
        criterion = nn.BCEWithLogitsLoss()
        metrics = evaluate_binary(model, test_loader, device, criterion)
        
        print("\n================ Evaluation Results (ANN) ================")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['acc']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print("----------------------------------------------------")
        print(f"True Positives (TP): {metrics['tp']}")
        print(f"False Positives (FP): {metrics['fp']}")
        print(f"True Negatives (TN): {metrics['tn']}")
        print(f"False Negatives (FN): {metrics['fn']}")
        print("====================================================")

if __name__ == "__main__":
    main()
