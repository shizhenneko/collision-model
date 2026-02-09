# 技术路线与训练实施指南 (Technical Route & Training Guide)

本文档补充了数据构建与训练实施的具体细节，特别是针对“多文件夹数据源”的训练方案。

## 1. 高可靠数据构建管线 (New Data Pipeline)

为了解决原有的时序断裂问题，我们引入了基于物理时间的构建脚本 `scripts/build_dataset.py`。

### 核心流程
1.  **分段 (Segmentation)**: 根据时间戳间隙（默认 >1.0s）将数据自动切分为独立的 Episodes。
2.  **时序打标 (Time-safe Labeling)**:
    *   在当前 Episode 内，基于**时间戳**查找未来 `t+0.1s`, `t+0.5s`, `t+1.0s` 的帧。
    *   即使中间有坏数据（如全黑图像），只要时间戳连续，我们依然能找到正确的“未来动作”作为标签。
    *   如果未来帧缺失（采集结束或中断），标签被标记为 `invalid`。
3.  **过滤 (Filtering)**:
    *   应用静止过滤、图像质量过滤、地板检测。
    *   **关键点**：过滤只删除当前帧，**不会影响**其他帧作为“未来标签”的有效性（因为打标步骤已在前一步完成）。
4.  **增强 (Augmentation)**: 对筛选出的高质量帧进行多模态同步增强（扩充 5-20 倍）。

### 使用方法
```bash
# 处理第一个采集文件夹
python scripts/build_dataset.py --input_dir datas/run1 --output_dir data_ready/run1 --aug_factor 5

# 处理第二个采集文件夹
python scripts/build_dataset.py --input_dir datas/run2 --output_dir data_ready/run2 --aug_factor 5

# 处理第三个采集文件夹
python scripts/build_dataset.py --input_dir datas/run3 --output_dir data_ready/run3 --aug_factor 5
```

---

## 2. 多文件夹训练实施方案 (Training with Multiple Folders)

针对您提出的“使用三个清洗过的文件夹进行训练”的需求，我们不需要物理合并它们，而是使用 PyTorch 的 `ConcatDataset` 进行逻辑拼接。

### 2.1 数据集类定义 (Dataset Class)

在您的训练脚本（如 `train.py`）中，需要定义一个支持绝对/相对路径修正的 Dataset 类。

```python
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import cv2
import numpy as np
import ast

class CollisionDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): 数据集的根目录（例如 'data_ready/run1'）
        """
        self.root_dir = root_dir
        self.csv_path = os.path.join(root_dir, 'data.csv')
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Data CSV not found at {self.csv_path}")
            
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} samples from {root_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load Image
        # 修正路径：CSV中存储的是 'images/xxx.jpg'，我们需要拼接 root_dir
        img_rel_path = row['image_path']
        img_full_path = os.path.join(self.root_dir, img_rel_path)
        
        image = cv2.imread(img_full_path)
        if image is None:
            # 容错处理：如果图片损坏，随机返回另一个样本（避免训练中断）
            return self.__getitem__(np.random.randint(0, len(self)))
            
        # Image Preprocessing (Resize, Normalize, CHW)
        image = cv2.resize(image, (160, 120))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # 2. Load Lidar
        lidar = np.array(ast.literal_eval(row['lidar_ranges'])).astype(np.float32)
        lidar = torch.from_numpy(lidar) / 8.0 # Normalize
        
        # 3. Load IMU
        imu = np.array(ast.literal_eval(row['imu_data'])).astype(np.float32)
        imu = torch.from_numpy(imu)
        
        # 4. Load Labels
        # VOA Target: [v, w]
        target_action = torch.tensor([row['linear_x'], row['angular_z']], dtype=torch.float32)
        
        # Collision Label (Medium Scale)
        collision_label = torch.tensor([row['label_medium']], dtype=torch.float32)
        
        # Multi-scale Targets (if needed)
        # ...
        
        return {
            'image': image,
            'lidar': lidar,
            'imu': imu,
            'action': target_action,
            'collision': collision_label
        }
```

### 2.3 安全的数据划分代码示例 (Safe Split Strategy)

由于数据存在时序相关性和增强副本，**严禁直接使用随机划分 (Random Split)**。必须基于 `episode_id` 并考虑样本平衡进行划分。

以下是一个**动态平衡划分**的函数示例，请在训练代码中使用：

```python
from torch.utils.data import Subset

def split_dataset_by_episode_balanced(dataset, val_ratio=0.2):
    """
    按 Episode 划分训练集和验证集，并保证验证集样本占比接近 val_ratio。
    
    Args:
        dataset: 必须是 ConcatDataset 或 CollisionDataset，且能够访问到底层的 df['episode_id']
        val_ratio: 验证集目标比例 (默认0.2)
        
    Returns:
        train_set, val_set (torch.utils.data.Subset)
    """
    # 1. 提取所有样本的 episode_id
    # 注意：如果是 ConcatDataset，需要遍历子数据集并加上偏移量以区分不同文件夹的相同ID
    # 这里为了简化，假设我们先处理好一个大的 DataFrame 或列表
    
    # 更加通用的做法是：在 Dataset __init__ 时加载所有 DataFrame 并合并
    # 这里演示针对单个 DataFrame 的逻辑，如果是 ConcatDataset，建议先合并 DataFrame 再实例化 Dataset
    
    if hasattr(dataset, 'df'):
        df = dataset.df
    else:
        # 如果是 ConcatDataset，可能需要手动聚合信息，或者在构建 Dataset 前先划分 CSV
        raise ValueError("Dataset must expose .df attribute for episode-based split")

    # 统计每个 episode 的长度
    ep_counts = df.groupby('episode_id').size().reset_index(name='count')
    
    # 随机打乱 Episode 列表
    ep_counts = ep_counts.sample(frac=1, random_state=42).reset_index(drop=True)
    
    total_samples = len(df)
    target_val_samples = int(total_samples * val_ratio)
    
    current_val_samples = 0
    val_episodes = []
    
    # 贪心累加，直到达到目标数量
    for _, row in ep_counts.iterrows():
        if current_val_samples < target_val_samples:
            val_episodes.append(row['episode_id'])
            current_val_samples += row['count']
        else:
            break
            
    # 生成索引掩码
    is_val = df['episode_id'].isin(val_episodes)
    val_indices = df.index[is_val].tolist()
    train_indices = df.index[~is_val].tolist()
    
    print(f"Split Result:")
    print(f"  Total Samples: {total_samples}")
    print(f"  Train Samples: {len(train_indices)} ({len(train_indices)/total_samples:.1%})")
    print(f"  Val Samples:   {len(val_indices)} ({len(val_indices)/total_samples:.1%})")
    print(f"  Val Episodes:  {sorted(val_episodes)}")
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices)
```

**最佳实践**：
建议先读取所有 CSV，合并为一个大的 DataFrame，进行上述划分得到 `train_df` 和 `val_df`，然后再分别实例化两个 `CollisionDataset`（传入 DataFrame 而不是目录），这样最为稳健。

