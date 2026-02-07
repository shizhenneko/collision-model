# VOA (Vision-Other-Navigation) 端到端多模态防碰撞系统方案 (Real Robot Onboard Edition)

鉴于您的硬件环境（真实机器人、VMware 无显卡用于控制/采集、WSL2 有显卡用于训练）以及核心需求（多模态感知、离线训练），本方案采用 **"VMware 采集数据 -> WSL2 离线训练 -> 机器人本地 CPU 推理"** 的架构。

## 1. 系统架构：离线训练与板载推理 (Offline Training & Onboard Inference)

### 1.1 硬件与环境定义
*   **机器人平台**: 真实移动机器人 (小车)。
*   **传感器配置**:
    *   **Camera**: 采集 RGB 图像 (视觉感知)。
    *   **Lidar**: 采集 360 度距离信息 (LaserScan, 360个点距数组)。
    *   **IMU**: 采集线速度 ($v_x, v_y, v_z$)、角速度 ($\omega_x, \omega_y, \omega_z$) 及地磁朝向 (惯性感知)。
*   **计算平台**:
    *   **控制与推理端 (VMware - Ubuntu 20.04)**: 运行 ROS Noetic，无 GPU。负责驱动硬件、数据采集、以及**加载模型进行本地实时推理** (CPU)。
    *   **训练端 (WSL2 - Ubuntu 22.04)**: 运行 PyTorch，有 GPU。仅负责繁重的离线模型训练任务。

### 1.2 工作流模式
1.  **Mode A: 数据采集 (Data Collection)**
    *   **运行端**: VMware (ROS)。
    *   **操作**: 人工遥控机器人，在真实环境中行走，覆盖安全行驶与（受控的）轻微碰撞/近碰撞场景。
    *   **输出**: 带有时间戳同步的离线数据集 (Rosbag 或 CSV+Image 序列)，包含所有传感器数据与人工控制指令。
2.  **Mode B: 离线训练 (Offline Training)**
    *   **运行端**: WSL2。
    *   **输入**: 手动导入 Mode A 采集的数据集。
    *   **任务**: 训练 VOA 策略网络 (模仿学习) 和 ANN 碰撞预测模型 (监督学习)。
3.  **Mode C: 板载推理 (Onboard Inference)**
    *   **运行端**: VMware (Local CPU)。
    *   **流程**: 将 WSL2 训练好的模型拷贝至 VMware -> 机器人加载模型 (PyTorch/ONNX) -> 本地 CPU 实时推理 -> 输出动作/碰撞概率 -> 底盘执行。

## 2. 网络架构详细设计 (Detailed Network Architecture)

为了适应本地 CPU 推理，网络设计必须在**轻量化**与**特征表达能力**之间取得平衡。同时，本系统采用**端到端学习**，无需人工对图像或雷达数据进行内容标注。

### 2.1 Vision Head (视觉感知层)
*   **数据标注**: **无需人工标注**。网络通过端到端训练，自动学习图像中对避障有用的特征。
*   **模型选型**: 首选 **ShuffleNetV2 0.5x** (针对 Yeahbot/ARM CPU 极致优化)。
    *   **备选**: **MobileNetV3-Small**。
    *   *选型分析*: 鉴于 Yeahbot 的弱 CPU 性能，ShuffleNetV2 0.5x 在 ARM 架构上通常比 MobileNetV3 快 20% 以上。**ResNet-18** 计算量过大，予以弃用。
*   **训练策略**: 使用 **ImageNet 预训练权重** (Transfer Learning) 初始化 Backbone，冻结浅层，仅微调深层，以适应小规模数据集。
*   **输入/输出**: RGB 图像 (Resize至160x120) -> CNN -> 256/512维特征向量。

### 2.2 Lidar Head (激光感知层)
*   **数据标注**: **无需人工标注**。直接使用原始距离数据。
*   **模型选型**: **1D Convolution (一维卷积)**。
    *   **关键技术**: 必须使用 **Circular Padding (环形填充)**。
        *   *原因*: 激光雷达 360 度数据首尾相接 ($0^\circ$ 与 $359^\circ$ 相邻)。普通 Padding 会导致边界断裂，Circular Padding 保证了全方位的几何特征连续性。
    *   结构: `Conv1d(k=5, padding_mode='circular') -> MaxPool -> ... -> Flatten`。
*   **输入/输出**: 360维归一化距离数组 -> 1D CNN -> 128维特征向量。

### 2.3 IMU Head (惯性感知层)
*   **数据标注**: **无需人工标注**。
*   **模型选型**: **Time Window + MLP**。
    *   *改进*: 单帧数据难以区分“正常减速”与“碰撞急停”。采用 **时间窗口 (Time Window)** 输入，捕捉动态趋势。
    *   **架构**: 取过去 $N$ 帧 (e.g., 0.5秒内 10 帧) 数据 -> **Flatten** -> **MLP**。相比 LSTM，Flatten+MLP 在弱 CPU 上推理速度最快。
*   **输入/输出**: $10 \times 9$ 维时序矩阵 -> Flatten -> MLP -> 64维特征向量。

### 2.4 Fusion Layer (多模态融合层)
*   **融合策略**: **Concatenation (拼接)**。
    *   $F_{fused} = Concat([F_{vision}, F_{lidar}, F_{imu}])$。
    *   总维度: $256 + 128 + 64 = 448$。
*   **特征交互**: 拼接后通过 1-2 层全连接层 (`Linear -> ReLU`)，生成最终的状态编码 (State Embedding)。

### 2.5 任务头与训练目标 (Task Heads & Labels)

| 任务头 | 模型结构 | 输出 | 训练方法 | 标签来源 (Label Source) |
| :--- | :--- | :--- | :--- | :--- |
| **VOA Policy** (策略) | MLP | $v, \omega$ | **模仿学习 (BC)** | **人工遥控指令** (ROS `/cmd_vel`)。网络学习拟合专家的驾驶动作。 |
| **ANN Collision** (碰撞) | MLP | 概率 $P$ | **监督学习** | **自动生成**。脚本分析未来 N 秒的 Lidar 数据，若距离 < 阈值则标为 1 (危险)，否则为 0。 |

## 3. 实施步骤规划 (Roadmap)

### 第一阶段：数据采集系统开发 (VMware)
1.  **Sensor Integration**: 编写/配置 ROS launch 文件，同时启动 Camera, Lidar, IMU 驱动节点。
2.  **Data Recorder Node**: 开发 Python 脚本，订阅所有传感器 topic (`/camera/image_raw`, `/scan`, `/imu`) 和控制指令 (`/cmd_vel`)。
    *   **同步 (Sync)**: 使用 `message_filters` 实现多传感器数据的时间戳对齐。
    *   **存储**: 按帧保存数据，例如 `image_{timestamp}.jpg` 和 `data.csv` (行内容: timestamp, lidar[360], imu[9], v, w)。

### 第二阶段：离线训练管道搭建 (WSL2)
1.  **Data Loader**: 编写 PyTorch Dataset 类，解析 VMware 采集的 CSV 和图像文件。
    *   **实现自动打标**: 在 `__getitem__` 中根据未来时刻的 Lidar 数据动态生成碰撞标签。
2.  **Model Implementation**: 搭建包含 Vision (MobileNet), Lidar (1D-CNN), IMU (MLP) 分支的多模态网络结构。
3.  **Training**:
    *   **Step 1**: 训练 **ANN 碰撞模型**，确保模型能从传感器数据中识别危险。
    *   **Step 2**: 训练 **VOA 策略网络**，基于 MSE Loss 学习专家的驾驶动作。

### 第三阶段：模型部署与本地推理 (Onboard Deployment)
1.  **Model Export (WSL2)**: **强制转换为 ONNX 格式**。
    *   PyTorch 原生推理在树莓派级别 CPU 上效率极低。
    *   必须使用 `torch.onnx.export` 将模型导出为标准 `.onnx` 文件。
2.  **Inference Node (VMware)**: 编写基于 **ONNX Runtime** 的 ROS 推理节点。
    *   **引擎**: 使用 `onnxruntime` (C++ 或 Python API) 进行推理，利用其 Graph Optimization 加速。
    *   加载模型文件 (`model.onnx`)。
    *   实时处理传感器数据并输入模型。
    *   直接发布控制指令到 `/cmd_vel`。
3.  **Safety Shield**: 在 VMware 端集成最终保护逻辑（如 ANN 预测高风险或 Lidar 检测极近障碍物时，无视网络指令强制刹车），保障系统安全。
