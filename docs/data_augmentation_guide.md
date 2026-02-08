# 数据增强与清洗指导文档 (VOA & ANN Multi-Modal Collision Avoidance)

**基于方案B：奖励加权回归 (Reward-Weighted Regression, RWR)**

---

## 目录

1. [概述](#概述)
2. [数据规模分析](#数据规模分析)
3. [基础数据清洗](#基础数据清洗)
4. [自动标签生成](#自动标签生成)
5. [数据增强](#数据增强)
6. [RWR方案实施](#rwr方案实施)
7. [完整训练流程](#完整训练流程)
8. [代码实现示例](#代码实现示例)

---

## 概述

### 核心目标

本项目采用**端到端多模态防碰撞系统**，训练两个关键模型：

| 模型 | 功能 | 训练方法 | 输出 |
|------|------|---------|------|
| **VOA Policy Network** | 策略网络，学习驾驶动作 | 奖励加权行为克隆 (RWR) | $v, \omega$ (线速度, 角速度) |
| **ANN Collision Model** | 碰撞预测网络，评估风险 | 监督学习 (二元分类) | $P_{crash} \in [0,1]$ (碰撞概率) |

### 方案选择：RWR (奖励加权回归)

**核心思想**：不直接剔除数据，而是给"安全"的数据更高权重，给"危险"的数据极低权重。

```
原始 Loss:  Loss = ||a_pred - a_expert||^2
加权 Loss:  Loss = exp(R / T) * ||a_pred - a_expert||^2
奖励定义:  R = 1 - P_crash
```

**优势**：
- 保留所有信息，避免误删好样本
- 软过滤，鲁棒性强
- 仅需调节温度参数T，调参简单

---

## 数据规模分析

### 当前数据状态

```
data.csv:      5,710 条记录
images:        5,713 张图片
预计清洗后:    ~2,500 条 (过滤静止数据)
```

### 数据量评估

| 模型类型 | 推荐数据量 | 当前数据量 | 缺口 | 解决方案 |
|---------|-----------|-----------|------|---------|
| VOA Policy Network | 50,000+ | ~2,500 | 95% | 数据增强 (20x扩充) |
| ANN Collision Model | 10,000+ | ~2,500 | 75% | 数据增强 (5x扩充) |

**结论**：原始数据量严重不足，必须通过数据增强扩充到50,000+条。

---

## 基础数据清洗

### 重要性

基础数据清洗是**物理层面的去噪**，无论采用方案A还是方案B，都必须执行。

### 清洗项目

#### 1. Lidar (360维距离数组)

```python
def clean_lidar(lidar_ranges):
    """
    Lidar数据清洗
    """
    lidar = np.array(lidar_ranges)

    # 1. 长度检查
    if len(lidar) != 360:
        return None

    # 2. 异常值过滤 (物理约束)
    # - 近端噪声: 距离 < 0.05m 的读数视为无效
    # - 远端噪声: 距离 > 8.0m 的读数视为无效
    lidar = np.clip(lidar, 0.05, 8.0)

    # 3. 异常尖峰检测 (可选)
    # 如果相邻点差异过大，可能是噪声
    for i in range(360):
        prev_idx = (i - 1) % 360
        next_idx = (i + 1) % 360

        if abs(lidar[i] - lidar[prev_idx]) > 2.0 and \
           abs(lidar[i] - lidar[next_idx]) > 2.0:
            lidar[i] = (lidar[prev_idx] + lidar[next_idx]) / 2

    return lidar.tolist()
```

#### 2. IMU (9维惯性数据)

```python
def estimate_imu_bias(imu_samples, num_samples=100):
    """
    估计IMU零偏（需要采集静止时的数据）
    """
    imu_samples = np.array(imu_samples[:num_samples])
    bias = np.mean(imu_samples, axis=0)
    return bias

def clean_imu(imu_data, bias):
    """
    IMU数据清洗
    """
    imu = imu_data - bias  # 零偏校正

    # 异常值滤除
    imu = np.clip(imu, -5.0, 5.0)  # 合理的线速度/角速度范围

    return imu.tolist()
```

#### 3. 图像 (RGB)

```python
def clean_image(image_path):
    """
    图像质量检查
    """
    img = cv2.imread(image_path)

    # 1. 文件存在性
    if img is None:
        return None

    # 2. 全黑检查
    if np.mean(img) < 10:
        return None

    # 3. 过曝检查
    if np.mean(img) > 250:
        return None

    # 4. 尺寸统一 (Resize到160x120)
    img = cv2.resize(img, (160, 120))

    # 5. 颜色空间统一 (BGR→RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
```

#### 4. 时间戳对齐

```python
def check_timestamp_sync(camera_ts, lidar_ts, imu_ts, cmd_vel_ts, tolerance=0.1):
    """
    检查多传感器时间戳是否同步
    """
    timestamps = [camera_ts, lidar_ts, imu_ts, cmd_vel_ts]

    for i in range(len(timestamps)):
        for j in range(i + 1, len(timestamps)):
            if abs(timestamps[i] - timestamps[j]) > tolerance:
                return False

    return True
```

#### 5. 静止数据过滤

```python
def filter_stationary_data(df, min_lin_vel=0.01, min_ang_vel=0.01):
    """
    过滤静止数据（机器人在等待或停止的样本）

    原理：
    - 行为克隆需要学习"如何运动"
    - 静止数据不包含运动决策信息
    - 保留条件: |v| > threshold OR |w| > threshold
    """
    is_moving = (df['linear_x'].abs() > min_lin_vel) | \
                (df['angular_z'].abs() > min_ang_vel)

    return df[is_moving].copy()
```

---

## 自动标签生成

### 原理

ANN Collision Model是**二元分类任务**，需要明确的标签：
- **y=1**: 危险（未来T秒内会发生碰撞）
- **y=0**: 安全（未来T秒内不会碰撞）

**关键思想**：利用Lidar的**客观距离测量**自动生成标签，无需人工标注。

### 实现方法

```python
def generate_auto_labels(df, future_window=10, collision_threshold=0.2):
    """
    自动生成碰撞标签

    物理意义：
    - 检查未来T秒（10帧≈0.5秒）内，Lidar最小距离
    - 如果存在距离 < 0.2m 的障碍物，标记为危险 (y=1)
    - 否则标记为安全 (y=0)

    Args:
        df: DataFrame包含lidar_ranges列
        future_window: 未来检查的帧数
        collision_threshold: 碰撞距离阈值 (m)

    Returns:
        labels: numpy数组，形状为[n_samples,]，值为0或1
    """
    n_samples = len(df)
    labels = np.zeros(n_samples, dtype=int)

    for t in range(n_samples):
        # 获取未来T帧的Lidar数据
        end_idx = min(t + future_window, n_samples)
        future_lidars = df.iloc[t:end_idx]['lidar_ranges']

        # 找出未来T帧内，Lidar检测到的最小距离
        min_distance = float('inf')

        for lidar in future_lidars:
            # 解析Lidar数据（如果是字符串格式）
            if isinstance(lidar, str):
                lidar = ast.literal_eval(lidar)

            # 找到当前帧的最小距离（忽略无穷大值）
            valid_ranges = [r for r in lidar if r < 8.0]
            if valid_ranges:
                min_dist_in_frame = min(valid_ranges)
                min_distance = min(min_distance, min_dist_in_frame)

        # 生成标签
        if min_distance < collision_threshold:
            labels[t] = 1  # 危险
        else:
            labels[t] = 0  # 安全

    return labels
```

### 直观理解

```
时间轴示例：
t=0s     t=0.05s   t=0.1s    t=0.5s
|--------|----------|---------|--------|
  Lidar   Lidar     Lidar    Lidar
  [3.0,   [2.5,     [1.0,    [0.1,   ← 0.1m < 0.2m，危险！
   2.5...] 2.0...]   0.5...]   0.2...]

自动标签生成：
- 对t=0s的数据：检查未来0.5s内Lidar，发现最小距离=0.1m < 0.2m
- 结论：标签y=1（危险）
```

---

## 数据增强

### 为什么需要数据增强？

**问题**：
- 原始数据：2,500条
- 需要数据：50,000+条
- 缺口：95%

**解决方案**：通过数据增强，每条原始样本扩充20倍，达到50,000条。

### 机器人场景的数据增强方法

#### 1. 图像增强 (Vision)

```python
import albumentations as A

image_transform = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),  # 亮度/对比度变化（适应不同光照）
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # 轻微模糊（模拟运动模糊）
    A.HorizontalFlip(p=0.5),  # 左右翻转（左右对称场景）
    A.Rotate(limit=15, p=0.3),  # 小角度旋转
])

def augment_image(image):
    return image_transform(image=image)['image']
```

**物理意义**：
- **亮度/对比度**：机器人可能在清晨、正午、黄昏行驶
- **高斯模糊**：摄像头可能有轻微抖动或运动模糊
- **左右翻转**：走廊等场景左右对称，增加样本多样性
- **小角度旋转**：模拟轻微的摄像头朝向偏差

#### 2. Lidar增强 (激光雷达)

```python
def augment_lidar(lidar_ranges):
    """
    Lidar 360度数据的环形增强

    原理：
    - Lidar是360度环形数据，首尾相接
    - 环形平移相当于"场景旋转"，保持几何一致性
    """
    lidar = np.array(lidar_ranges)

    # 1. 环形平移（相当于场景旋转）
    shift = random.randint(0, 180)  # 平移0-180度
    augmented = np.roll(lidar, shift)

    # 2. 添加高斯噪声（模拟传感器噪声）
    noise = np.random.normal(0, 0.02, len(lidar))
    augmented += noise

    # 3. 重新裁剪到合理范围
    augmented = np.clip(augmented, 0.05, 8.0)

    return augmented.tolist()
```

**物理意义**：
- **环形平移**：机器人可以360度旋转，平移模拟不同朝向
- **添加噪声**：激光雷达有固有测量噪声，提高模型鲁棒性

#### 3. IMU增强 (惯性传感器)

```python
def augment_imu(imu_data):
    """
    IMU数据增强（主要是添加噪声）

    IMU数据: [v_x, v_y, v_z, w_x, w_y, w_z, a_x, a_y, a_z]
    - v: 线速度
    - w: 角速度
    - a: 加速度（磁力计数据）
    """
    imu = np.array(imu_data)

    # 添加轻微的高斯噪声
    noise = np.random.normal(0, 0.001, len(imu))
    imu += noise

    return imu.tolist()
```

#### 4. 多模态协同增强（关键！）

```python
def augment_multimodal_sample(sample):
    """
    多模态协同增强

    核心原则：
    - 多个传感器感知同一场景
    - 增强必须同步，否则传感器数据错配
    """
    augmented = sample.copy()

    # 图像左右翻转
    if random.random() < 0.5:
        # 1. 图像翻转
        augmented['image'] = cv2.flip(augmented['image'], 1)

        # 2. 同步翻转Lidar（必须！）
        # 因为视觉+激光必须对齐同一场景
        augmented['lidar'] = np.flip(augmented['lidar'])

        # 3. 同步调整角速度方向
        # 左翻后，原本的左转变成右转
        augmented['angular_z'] = -augmented['angular_z']

    # Lidar环形平移
    if random.random() < 0.5:
        shift = random.randint(0, 360)
        augmented['lidar'] = np.roll(augmented['lidar'], shift)

    # 添加传感器噪声
    augmented['lidar'] += np.random.normal(0, 0.02, 360)
    augmented['imu'] += np.random.normal(0, 0.001, 9)

    return augmented
```

**为什么必须同步？**

```
错误示例：
- 图像左翻（原本左边有墙 → 右边有墙）
- Lidar不翻（左边仍然检测到墙）

结果：视觉看到"右边有墙"，Lidar看到"左边有墙"
后果：多模态融合学到错误的关联，严重损害模型性能

正确做法：
- 图像左翻 + Lidar同步左翻 + 角速度取反
- 保持多模态数据物理一致性
```

---

## RWR方案实施

### 核心流程

```
Step 1: 基础数据清洗（物理层面）
   ↓
Step 2: 自动标签生成（基于未来Lidar）
   ↓
Step 3: 训练ANN Collision Model
   ↓
Step 4: 用ANN对所有数据打分（不清洗！）
   ↓
Step 5: 训练VOA Policy（用加权Loss）
```

### 详细步骤

#### Step 1: 基础数据清洗

```python
def basic_cleaning(raw_data):
    """
    基础数据清洗（物理层面）

    作用：
    - Lidar异常值过滤
    - IMU零偏校正
    - 图像质量检查
    - 时间戳对齐
    - 静止数据过滤
    """
    cleaned_data = []

    for sample in raw_data:
        # 1. 时间戳同步检查
        if not check_timestamp_sync(sample['timestamps']):
            continue

        # 2. Lidar清洗
        lidar = clean_lidar(sample['lidar'])
        if lidar is None:
            continue

        # 3. IMU清洗
        imu = clean_imu(sample['imu'], imu_bias)
        if imu is None:
            continue

        # 4. 图像清洗
        image = clean_image(sample['image_path'])
        if image is None:
            continue

        # 5. 构建清洗后的样本
        cleaned_sample = {
            'timestamp': sample['timestamp'],
            'image': image,
            'image_path': sample['image_path'],
            'lidar': lidar,
            'imu': imu,
            'linear_x': sample['linear_x'],
            'angular_z': sample['angular_z']
        }

        cleaned_data.append(cleaned_sample)

    # 6. 静止数据过滤
    df = pd.DataFrame(cleaned_data)
    df = filter_stationary_data(df, min_lin_vel=0.01, min_ang_vel=0.01)

    return df
```

#### Step 2: 自动标签生成

```python
def step2_auto_label_generation(cleaned_df):
    """
    生成ANN训练用的标签
    """
    labels = generate_auto_labels(
        cleaned_df,
        future_window=10,      # 未来10帧（约0.5秒）
        collision_threshold=0.2 # 碰撞距离阈值0.2m
    )

    # 统计标签分布
    n_danger = np.sum(labels)
    n_safe = len(labels) - n_danger
    print(f"标签分布: 安全={n_safe}, 危险={n_danger}, "
          f"危险率={n_danger/len(labels):.2%}")

    return labels
```

#### Step 3: 训练ANN Collision Model

```python
def train_ann_model(dataset, labels, num_epochs=50):
    """
    训练ANN碰撞预测模型

    ANN结构：
    - 输入: 多模态融合特征 (448维)
    - 隐藏层: MLP(448 -> 256 -> 128)
    - 输出: 碰撞概率 P_crash (1维)
    """
    model = ANNCollisionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch in dataset:
            # 前向传播
            state = batch['state']  # 多模态特征
            label = batch['label']

            P_crash_pred = model(state)

            # 计算损失（二元交叉熵）
            loss = criterion(P_crash_pred, label)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    return model
```

#### Step 4: 用ANN打分（不清洗！）

```python
def step4_ann_scoring(ann_model, dataset):
    """
    用ANN对所有数据打分

    关键点：
    - 不剔除任何样本（方案B的核心）
    - 保留所有数据，仅用于计算奖励权重
    """
    rewards = []

    for sample in dataset:
        # 前向传播ANN
        with torch.no_grad():
            P_crash = ann_model(sample['state'])

        # 计算奖励: R = 1 - P_crash
        # 越安全，奖励越高
        R = 1.0 - P_crash.item()

        rewards.append(R)

    # 统计奖励分布
    rewards = np.array(rewards)
    print(f"奖励分布: min={rewards.min():.3f}, max={rewards.max():.3f}, "
          f"mean={rewards.mean():.3f}")

    return rewards
```

#### Step 5: 训练VOA Policy（加权Loss）

```python
class RewardWeightedVOATrainer:
    def __init__(self, voa_model, T=1.0):
        """
        Args:
            voa_model: VOA策略网络
            T: 温度参数，控制权重尖锐程度
               - T=0.5: 更激进（安全样本权重是危险的4倍）
               - T=1.0: 标准（安全样本权重是危险的2倍）
               - T=2.0: 更温和（安全样本权重是危险的1.4倍）
        """
        self.voa_model = voa_model
        self.T = T
        self.optimizer = torch.optim.Adam(voa_model.parameters(), lr=0.0001)
        self.mse_criterion = nn.MSELoss()

    def train_epoch(self, dataset, rewards):
        """
        训练VOA策略网络（使用奖励加权Loss）

        核心公式：
            Loss = exp(R / T) * MSE(a_pred, a_expert)
        """
        total_loss = 0.0

        for batch in dataset:
            # 1. 前向传播VOA
            state = batch['state']
            action_pred = self.voa_model(state)

            # 2. 获取专家动作
            action_expert = batch['action']

            # 3. 计算MSE Loss
            mse_loss = self.mse_criterion(action_pred, action_expert)

            # 4. 获取奖励权重
            R = batch['reward']
            weight = torch.exp(torch.tensor(R / self.T))

            # 5. 奖励加权（关键步骤）
！            weighted_loss = weight * mse_loss

            # 6. 反向传播
            self.optimizer.zero_grad()
            weighted_loss.backward()
            self.optimizer.step()

            total_loss += weighted_loss.item()

        avg_loss = total_loss / len(dataset)

        return avg_loss

    def train(self, dataset, rewards, num_epochs=100):
        """
        完整训练循环
        """
        for epoch in range(num_epochs):
            loss = self.train_epoch(dataset, rewards)

            if epoch % 10 == 0:
                print(f"VOA Epoch {epoch}, Loss: {loss:.4f}")

        return self.voa_model
```

---

## 完整训练流程

### 主流程代码

```python
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

def main_rwr_pipeline():
    """
    RWR方案完整训练流程
    """
    print("=" * 60)
    print("RWR (Reward-Weighted Regression) 训练流程")
    print("=" * 60)

    # ===== Step 0: 加载原始数据 =====
    print("\n[Step 0] 加载原始数据...")
    raw_data = load_raw_csv('datas/data.csv')
    print(f"原始数据量: {len(raw_data)} 条")

    # ===== Step 1: 基础数据清洗 =====
    print("\n[Step 1] 基础数据清洗（物理层面）...")
    cleaned_df = basic_cleaning(raw_data)
    print(f"清洗后数据量: {len(cleaned_df)} 条")

    # ===== Step 2: 自动标签生成 =====
    print("\n[Step 2] 自动标签生成（基于未来Lidar）...")
    labels = step2_auto_label_generation(cleaned_df)

    # ===== Step 2.5: 数据增强 =====
    print("\n[Step 2.5] 数据增强...")
    augment_factor = 20  # 每条样本扩充到20条
    augmented_dataset = augment_dataset(cleaned_df, augment_factor)
    print(f"增强后数据量: {len(augmented_dataset)} 条")

    # ===== Step 3: 训练ANN Collision Model =====
    print("\n[Step 3] 训练ANN Collision Model...")
    ann_dataset = prepare_ann_dataset(augmented_dataset, labels)
    ann_model = train_ann_model(ann_dataset, labels, num_epochs=50)

    # ===== Step 4: 用ANN打分（不清洗！） =====
    print("\n[Step 4] 用ANN打分（计算奖励权重）...")
    rewards = step4_ann_scoring(ann_model, augmented_dataset)

    # ===== Step 5: 训练VOA Policy（加权Loss） =====
    print("\n[Step 5] 训练VOA Policy（奖励加权回归）...")
    voa_model = VOAPolicyNetwork()
    voa_trainer = RewardWeightedVOATrainer(voa_model, T=1.0)

    voa_dataset = prepare_voa_dataset(augmented_dataset, rewards)
    trained_voa = voa_trainer.train(voa_dataset, rewards, num_epochs=100)

    # ===== Step 6: 保存模型 =====
    print("\n[Step 6] 保存模型...")
    torch.save(ann_model.state_dict(), 'models/ann_collision_model.pth')
    torch.save(trained_voa.state_dict(), 'models/voa_policy_model.pth')
    print("模型已保存！")

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


def augment_dataset(base_df, augment_factor):
    """
    数据增强主函数
    """
    augmented_samples = []

    for idx in tqdm(range(len(base_df)), desc="数据增强"):
        base_sample = base_df.iloc[idx]

        for _ in range(augment_factor):
            # 多模态协同增强
            augmented = augment_multimodal_sample(base_sample.to_dict())
            augmented_samples.append(augmented)

    return augmented_samples
```

---

## 代码实现示例

### 完整的PyTorch Dataset类

```python
import torch
from torch.utils.data import Dataset
import cv2
import ast

class MultiModalDataset(Dataset):
    """
    多模态数据集类

    功能：
    1. 加载图像、Lidar、IMU数据
    2. 数据归一化
    3. 返回多模态融合特征
    """
    def __init__(self, samples, mode='voa'):
        """
        Args:
            samples: 样本列表
            mode: 'ann' 或 'voa'
                  - 'ann': 返回标签（用于ANN训练）
                  - 'voa': 返回专家动作和奖励（用于VOA训练）
        """
        self.samples = samples
        self.mode = mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. 加载图像
        image = self._load_image(sample['image_path'])
        if image is None:
            # 如果图像加载失败，返回下一个样本
            return self.__getitem__((idx + 1) % len(self))

        # 2. 加载Lidar
        lidar = self._load_lidar(sample['lidar'])

        # 3. 加载IMU
        imu = self._load_imu(sample['imu'])

        # 4. 多模态特征融合
        state = self._fuse_features(image, lidar, imu)

        # 5. 根据模式返回不同数据
        if self.mode == 'ann':
            # ANN训练：返回特征和标签
            return {
                'state': state,
                'label': sample['label']
            }
        elif self.mode == 'voa':
            # VOA训练：返回特征、专家动作和奖励
            action = torch.tensor([
                sample['linear_x'],
                sample['angular_z']
            ], dtype=torch.float32)

            reward = torch.tensor(sample['reward'], dtype=torch.float32)

            return {
                'state': state,
                'action': action,
                'reward': reward
            }

    def _load_image(self, image_path):
        """加载并预处理图像"""
        image = cv2.imread(image_path)
        if image is None:
            return None

        # Resize
        image = cv2.resize(image, (160, 120))
        # BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 归一化到[0, 1]
        image = image.astype(np.float32) / 255.0
        # HWC → CHW
        image = np.transpose(image, (2, 0, 1))
        # 转为Tensor
        image = torch.from_numpy(image)

        return image

    def _load_lidar(self, lidar_data):
        """加载并预处理Lidar数据"""
        if isinstance(lidar_data, str):
            lidar = np.array(ast.literal_eval(lidar_data))
        else:
            lidar = np.array(lidar_data)

        # 归一化到[0, 1]
        lidar = lidar / 8.0

        # 转为Tensor
        lidar = torch.from_numpy(lidar.astype(np.float32))

        return lidar

    def _load_imu(self, imu_data):
        """加载并预处理IMU数据"""
        if isinstance(imu_data, str):
            imu = np.array(ast.literal_eval(imu_data))
        else:
            imu = np.array(imu_data)

        # 归一化（假设合理范围是[-2, 2]）
        imu = imu / 2.0

        # 转为Tensor
        imu = torch.from_numpy(imu.astype(np.float32))

        return imu

    def _fuse_features(self, image, lidar, imu):
        """
        多模态特征融合

        注意：这里只是简单拼接
        实际网络中会使用Gated Fusion
        """
        # Vision Head: CNN提取特征
        # (假设已经预处理，这里简化处理)
        vision_feat = image.flatten()[:256]  # 取前256维作为示例

        # Lidar Head: 360维
        lidar_feat = lidar[:128]  # 取前128维

        # IMU Head: 9维
        imu_feat = imu[:64]  # 取前64维（示例）

        # 拼接
        fused = torch.cat([vision_feat, lidar_feat, imu_feat])

        return fused
```

### 使用示例

```python
# ===== 创建数据集 =====
ann_dataset = MultiModalDataset(augmented_samples, mode='ann')
voa_dataset = MultiModalDataset(augmented_samples, mode='voa')

# ===== 创建数据加载器 =====
batch_size = 32
ann_loader = torch.utils.data.DataLoader(
    ann_dataset, batch_size=batch_size, shuffle=True
)
voa_loader = torch.utils.data.DataLoader(
    voa_dataset, batch_size=batch_size, shuffle=True
)

# ===== 训练ANN =====
for epoch in range(50):
    for batch in ann_loader:
        state = batch['state']
        label = batch['label']

        P_crash_pred = ann_model(state)
        loss = criterion(P_crash_pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ===== 训练VOA =====
for epoch in range(100):
    for batch in voa_loader:
        state = batch['state']
        action_expert = batch['action']
        reward = batch['reward']

        action_pred = voa_model(state)
        mse_loss = criterion(action_pred, action_expert)
        weight = torch.exp(reward / T)
        weighted_loss = weight * mse_loss

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
```

---

## 总结

### 核心要点

| 要点 | 说明 |
|-----|------|
| **基础清洗** | Lidar/IMU/图像异常值过滤、时间戳对齐、静止数据过滤 |
| **自动标签** | 基于未来Lidar距离自动生成碰撞标签 (y=0/1) |
| **数据增强** | 图像/Lidar/IMU多模态协同增强，20x扩充样本 |
| **ANN训练** | 二元分类，预测碰撞概率 P_crash |
| **RWR方案** | 用ANN计算奖励，对VOA训练Loss进行指数加权 |
| **温度参数T** | 控制权重尖锐程度 (T=0.5激进, T=1.0标准, T=2.0温和) |

### 流程图

```
原始数据 (5,710条)
    ↓
[基础清洗] 物理去噪
    ↓
清洗后数据 (~2,500条)
    ↓
[自动标签] 基于未来Lidar
    ↓
[数据增强] 20x扩充
    ↓
增强后数据 (~50,000条)
    ↓
┌─────────────┬─────────────┐
│ ANN训练     │   VOA训练   │
├─────────────┼─────────────┤
│ 二元分类    │ 加权回归    │
│ 输入: P_crash│ 权重: exp(R/T)│
│ 输出: 0/1  │ R = 1-P_crash│
└─────────────┴─────────────┘
```

### 参考文档

- `plan.md`: 系统架构与网络设计
- `data_cleaning.md`: 数据清洗策略详解
- `training_supplement.md`: RWR方案原理

---

**文档版本**: v1.0
**创建日期**: 2026-02-08
**适用方案**: 方案B (奖励加权回归 RWR)
