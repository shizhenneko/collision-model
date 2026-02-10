import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from dataset import CollisionDataset, split_dataset_train_val_test
from models import CollisionModel
import torchvision.transforms as T
import time

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        # 注意：这里需要浅拷贝，避免修改原 dataset 的缓存（如果有的话）
        # 但 CollisionDataset 每次 __getitem__ 都是重新读取，所以安全
        sample = self.dataset[idx]
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample

def evaluate_binary(model, loader, device, criterion, threshold=0.5):
    model.eval()
    total_loss = 0.0
    total_count = 0
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            lidars = batch['lidar'].to(device)
            imus = batch['imu'].to(device)
            labels = batch['collision'].to(device)

            logits = model(images, lidars, imus)
            loss = criterion(logits, labels)
            outputs = torch.sigmoid(logits)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            preds = (outputs >= threshold).float()
            labels_bool = (labels >= 0.5).float()

            tp += int(((preds == 1) & (labels_bool == 1)).sum().item())
            fp += int(((preds == 1) & (labels_bool == 0)).sum().item())
            tn += int(((preds == 0) & (labels_bool == 0)).sum().item())
            fn += int(((preds == 0) & (labels_bool == 1)).sum().item())

    avg_loss = total_loss / max(total_count, 1)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)

    return {
        'loss': avg_loss,
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

def main():
    # ================= 配置参数 (Configuration) =================
    # 数据集路径列表
    data_roots = [
        '../data_ready',
        '../data_ready_1',
    ]
    
    batch_size = 64
    learning_rate = 1e-4
    num_epochs = 25
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ================= 数据准备 (Data Preparation) =================
    datasets = []
    for root in data_roots:
        # 检查路径是否存在
        full_path = os.path.abspath(root)
        if os.path.exists(full_path):
            print(f"Found dataset at: {full_path}")
            try:
                ds = CollisionDataset(full_path)
                datasets.append(ds)
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
        else:
            print(f"Warning: Dataset path not found: {full_path}")
            
    if not datasets:
        print("No valid datasets found. Exiting.")
        return

    # 合并数据集
    full_dataset = ConcatDataset(datasets)
    
    train_set, val_set, test_set = split_dataset_train_val_test(full_dataset, val_ratio=0.15, test_ratio=0.15, min_train_samples=1000)
    
    # 数据增强 (Data Augmentation)
    # 增加图像变换以防止过拟合
    train_transform = T.Compose([
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])
    train_set = TransformedDataset(train_set, transform=train_transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # ================= 模型构建 (Model Building) =================
    # 启用 Dropout (0.5) 防止过拟合
    model = CollisionModel(dropout=0.5).to(device)
    
    # ================= 优化器与损失函数 (Optimizer & Loss) =================
    criterion = nn.BCEWithLogitsLoss()
    # 增加 weight_decay (L2 正则化)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # ================= 训练循环 (Training Loop) =================
    best_val_loss = float('inf')
    save_dir = '../checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    print("Start Training ANN (Collision Model)...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for batch in train_loader:
            # 搬运数据到 GPU
            images = batch['image'].to(device)
            lidars = batch['lidar'].to(device)
            imus = batch['imu'].to(device)
            labels = batch['collision'].to(device) # [B, 1]
            
            optimizer.zero_grad()
            
            # 前向传播
            logits = model(images, lidars, imus) # [B, 1]
            
            # 计算损失
            loss = criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
            # 计算准确率 (阈值 0.5)
            outputs = torch.sigmoid(logits)
            preds = (outputs > 0.5).float()
            train_acc += (preds == labels).sum().item()
            
        train_loss /= len(train_set)
        train_acc /= len(train_set)
        
        # --- Validation Phase ---
        val_metrics = evaluate_binary(model, val_loader, device, criterion, threshold=0.5)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['acc']
        
        end_time = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Time: {end_time - start_time:.1f}s "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_metrics['f1']:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, 'ann_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  -> Model saved to {save_path}")

    print("Training Completed.")
    best_path = os.path.join(save_dir, 'ann_best.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_metrics = evaluate_binary(model, test_loader, device, criterion, threshold=0.5)
        print(
            "Test Metrics: "
            f"Loss={test_metrics['loss']:.4f} "
            f"Acc={test_metrics['acc']:.4f} "
            f"Precision={test_metrics['precision']:.4f} "
            f"Recall={test_metrics['recall']:.4f} "
            f"F1={test_metrics['f1']:.4f} "
            f"Confusion(TP/FP/TN/FN)={test_metrics['tp']}/{test_metrics['fp']}/{test_metrics['tn']}/{test_metrics['fn']}"
        )

if __name__ == '__main__':
    main()
