# 数据清洗与预处理策略（VOA/ANN 多模态防碰撞项目）

本文件明确：是否需要数据清洗、如何进行数据清洗、为什么要清洗。内容基于现有方案与补充文档，结合端到端训练、自动打标与多模态传感器特性进行整理。

## 概述
- 采集流程：VMware 上 ROS 节点采集 Camera/Lidar/IMU 与 `/cmd_vel`，形成带时间戳的数据集；在 WSL2 上进行离线训练；模型部署到机器人本地 CPU 推理。
- 端到端训练无需人工内容标注，但并不意味着可以使用“脏数据”。干净、对齐、稳定的数据是训练成功的前提。
- 参考： [plan.md](file:///c:/Users/86159/Desktop/collision-model/plan.md#L32-L41)（端到端、无需人工标注）；[plan.md](file:///c:/Users/86159/Desktop/collision-model/plan.md#L79-L81)（自动打标思路）；[training_supplement.md](file:///c:/Users/86159/Desktop/collision-model/training_supplement.md#L52-L57)（基于 ANN 的数据清洗）。

## 结论：需要数据清洗
- 必须进行“两层清洗”与“一层门控”的立体式质量控制：
  - **基础层清洗 (Hard)**：物理层面的去噪，剔除传感器掉线、时间戳错乱、严重损坏的“垃圾数据”。
  - **训练层清洗 (Hard)**：语义层面的筛选，利用 ANN 风险模型剔除专家操作失误的“负样本”。
  - **门控融合 (Soft)**：架构层面的自适应，利用 **Gated Fusion** 网络自动降低低质量（如暗光、模糊）但可用的数据权重，减少对完美数据的依赖。
- “无需人工标注”仅指不需要人为在图像/激光上画框或标记类别；但数据的时序对齐、异常剔除、标签自动生成与质量筛选仍然必要。

## 基础清洗与预处理
- 时间戳同步与完整性检查：对齐 Camera/Lidar/IMU 与 `/cmd_vel` 的时间戳；剔除缺帧或不同步样本；确保一帧内多模态数据一致。
- 图像（RGB）：
  - 尺寸与格式：统一 Resize 到 160×120；确保颜色空间一致（RGB）。
  - 质量过滤：剔除全黑/过曝/严重模糊帧；必要时做亮度归一化。
  - 归一化：按通道标准化到固定分布，减少光照差异影响。
- 激光（360 距离）：
  - 归一化与范围裁剪：限定合理量程（例如近端 <0.05m、远端 >8m 的异常读数剔除或截断）。
  - 环形一致性：检查首尾角度连续性；保留环形特征，避免边界断裂。
  - 异常值处理：过滤明显不合理的尖峰/饱和读数。
- IMU（线/角速度与磁姿）：
  - 零偏与漂移校正：静止采样估计零偏并扣除；对明显尖峰进行滤除。
  - 时间窗口化：按训练设计将单帧拓展为 N 帧窗口（例如 0.5s/10 帧），保持与标签的时序对齐。
- 数据组织：
  - 一帧一行：timestamp, image_path, lidar[360], imu[9], v, w。
  - 健康检查：输出统计（缺帧率、异常比、不同步样本数），便于审计。

## 训练集质量清洗（策略网络）
- 目的：避免策略网络学习到“危险情境下的有害动作”，提升安全性与泛化。
- 步骤：
  1. 先训练 ANN 碰撞模型，使其能给每帧状态打出风险分数 \(P_{crash} \in [0,1]\)（详见补充文档）。
  2. 对全数据集逐帧打分。
  3. 两种做法：
     - Safety-Filtered BC（推荐入门）：剔除 \(P_{crash} > 0.3\) 的样本，仅用“安全帧”训练策略网络。（阈值可按数据分布微调）
     - Reward-Weighted Regression（进阶）：不剔除，使用 \(R=1-P_{crash}\) 作为加权，令损失 \(Loss = e^{(R/T)} \cdot \| \hat{a}-a_{expert}\|^2\)，强调安全样本、弱化危险样本。
- 保留负样本用于 ANN 训练：对 ANN 本身的分类训练，危险样本（负样本）必须保留以维持类别平衡；清洗主要作用于策略网络的训练集。
- 参考： [training_supplement.md](file:///c:/Users/86159/Desktop/collision-model/training_supplement.md#L52-L57)、[training_supplement.md](file:///c:/Users/86159/Desktop/collision-model/training_supplement.md#L63-L68)。

## 标签生成与对齐
- 自动标签思路：依据某时刻后 \(T\) 秒内的 Lidar 距离是否小于阈值（如 0.2m）生成二元标签（危险/安全）。
- 对齐原则：策略输入的状态与用于生成该状态标签的“未来窗口”索引严格对应；避免索引越界与错配。
- 参考： [plan.md](file:///c:/Users/86159/Desktop/collision-model/plan.md#L79-L81)、[training_supplement.md](file:///c:/Users/86159/Desktop/collision-model/training_supplement.md#L33-L37)。

## 为什么要清洗
- 提升泛化与稳定性：减少噪声与异常样本对模型的干扰，降低过拟合风险。
- 强化安全性：防止策略网络学习到风险场景下的错误行为，减小部署时的危险动作概率。
- 保障时序一致性：多模态必须对齐，否则融合特征失真，训练目标与输入错配。
- 改善标签质量：自动打标依赖规则与阈值，清洗可减少“错误标签”引入的偏差。
- 便于分析与调优：清洗与审计能暴露数据采集中的系统性问题（传感器故障、同步不稳定）。

## 阈值与实践建议
- 初始阈值：\(P_{crash} = 0.3\) 作为过滤门限，后续可根据验证集表现调整。
- 可视化审计：统计被剔除样本比例、不同场景下的分布；必要时分场景设定不同阈值。
- 类别/场景平衡：在 ANN 训练中保证危险/安全样本的数量与场景多样性。
- 日志与复现：记录清洗规则、阈值、版本号与数据快照，确保可复现与可回滚。
- 渐进式调优：先用保守阈值，观察部署效果，再逐步放宽以提高效率。

## 参考文档
- [plan.md](file:///c:/Users/86159/Desktop/collision-model/plan.md#L32-L41)（端到端、无需人工标注）
- [plan.md](file:///c:/Users/86159/Desktop/collision-model/plan.md#L79-L81)（自动打标实现位置）
- [training_supplement.md](file:///c:/Users/86159/Desktop/collision-model/training_supplement.md#L52-L57)（基于 ANN 的清洗步骤）
- [training_supplement.md](file:///c:/Users/86159/Desktop/collision-model/training_supplement.md#L63-L68)（加权回归公式）

