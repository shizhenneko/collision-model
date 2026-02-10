import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

def _activation(name):
    name = (name or 'silu').lower()
    if name == 'silu':
        return nn.SiLU(inplace=True)
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'hardswish':
        return nn.Hardswish(inplace=True)
    if name == 'identity' or name == 'none':
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")

class ShuffleNetV2Ext(nn.Module):
    def __init__(self):
        super(ShuffleNetV2Ext, self).__init__()
        # 加载预训练的 ShuffleNetV2 0.5x
        # 注意：默认 weights=None, 如果需要预训练权重可以在训练脚本中指定或在这里加载
        # 这里我们假设用户会有网络连接下载权重，或者在训练时加载
        try:
            self.backbone = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
        except Exception:
            self.backbone = models.shufflenet_v2_x0_5(weights=None)
        
        # 移除最后的分类层 (fc)
        # ShuffleNetV2 0.5x 的 stage4 输出通道是 1024 (conv5) -> mean pool -> 1024
        # 我们只取特征提取部分
        # 原始 forward: x -> conv1 -> maxpool -> stage2 -> stage3 -> stage4 -> conv5 -> mean -> fc
        
        # 我们截取到 conv5 之后的 mean pooling 之前，或者包含 mean pooling
        # 为了灵活性，我们重写 forward
        del self.backbone.fc
        
    def forward(self, x):
        # x: [B, 3, 120, 160]
        x = self.backbone.conv1(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.stage2(x)
        x = self.backbone.stage3(x)
        x = self.backbone.stage4(x)
        x = self.backbone.conv5(x) # [B, 1024, 4, 5] (approx)
        
        # Global Average Pooling
        # 使用 AdaptiveAvgPool2d 替代 mean()，避免 ONNX 导出时的 ReduceMean 算子版本转换问题
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return x

class VisionEncoder(nn.Module):
    def __init__(self, output_dim=256, freeze_shallow=True, projector_hidden_dim=512, activation='silu', dropout=0.0):
        super(VisionEncoder, self).__init__()
        self.backbone = ShuffleNetV2Ext()
        
        # 冻结浅层 (Stage 1 & Stage 2)
        if freeze_shallow:
            # 冻结所有层
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # 开放 Stage 3
            for param in self.backbone.backbone.stage3.parameters():
                param.requires_grad = True
            
            # 开放 Stage 4
            for param in self.backbone.backbone.stage4.parameters():
                param.requires_grad = True
                
            # 开放 Conv5
            for param in self.backbone.backbone.conv5.parameters():
                param.requires_grad = True
                
        # 降维到 256
        self.projector = nn.Sequential(
            nn.Linear(1024, projector_hidden_dim),
            nn.LayerNorm(projector_hidden_dim),
            _activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(projector_hidden_dim, output_dim)
        )
        
    def forward(self, x):
        feat = self.backbone(x)
        return self.projector(feat)

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
        # 回归最原始的 torch.max 写法
        # 在 Opset 14 中，ReduceMax 支持 axes 输入，完全兼容

    def forward(self, x):
        # x: [B, C, L] -> [B, C, 1]
        # 使用 MaxPool1d(kernel_size=45) 替代 torch.max()
        # 避免 ONNX 导出时的 ReduceMax 算子版本转换问题
        # LidarEncoder 经过3次 stride=2 的卷积，360 -> 180 -> 90 -> 45
        return F.max_pool1d(x, kernel_size=45)

class LidarEncoder(nn.Module):
    def __init__(self, output_dim=128, activation='silu', dropout=0.0):
        super(LidarEncoder, self).__init__()
        # Input: [B, 1, 360]
        # Circular Padding 保持环形连续性
        # 增加通道数以提取更丰富的特征 (16/32/64 -> 32/64/128)
        self.net = nn.Sequential(
            # Conv1
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2, padding_mode='circular'),
            nn.BatchNorm1d(32),
            _activation(activation), 
            
            # Conv2
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(64),
            _activation(activation), 
            
            # Conv3
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(128),
            _activation(activation), 
            
            # Global Max Pooling
            GlobalMaxPool1d(),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, output_dim)
            # 移除 ReLU，允许输出负值
        )

    def forward(self, x):
        # x: [B, 360] -> [B, 1, 360]
        if x.dim() == 2:
            # 使用 reshape 替代 unsqueeze(1)，解决 Opset 11 导出时
            # axes 从 Input 转 Attribute 的 converter bug
            # 这种修改是数学等价的，不需要重新训练
            x = x.reshape(x.shape[0], 1, x.shape[1])
        feat = self.net(x)
        return self.fc(feat)

class ImuEncoder(nn.Module):
    def __init__(self, input_dim=9, output_dim=64, activation='silu', dropout=0.0):
        super(ImuEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            _activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, output_dim)
            # 移除 ReLU，允许输出负值
        )
        
    def forward(self, x):
        # x: [B, 9]
        return self.net(x)

class GatedFusion(nn.Module):
    def __init__(self, vis_dim=256, lidar_dim=128, imu_dim=64, gate_mode='softmax'):
        super(GatedFusion, self).__init__()
        self.total_dim = vis_dim + lidar_dim + imu_dim
        self.gate_mode = gate_mode
        
        # 门控网络：根据拼接后的特征计算每个模态的权重
        self.gate_net = nn.Sequential(
            nn.Linear(self.total_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3) # 3个模态的权重
        )
        
        self.vis_dim = vis_dim
        self.lidar_dim = lidar_dim
        self.imu_dim = imu_dim

    def forward(self, vis, lidar, imu):
        # 拼接
        concat_feat = torch.cat([vis, lidar, imu], dim=1)
        
        # 计算权重 [B, 3]
        gate_logits = self.gate_net(concat_feat)
        if self.gate_mode == 'softmax':
            weights = torch.softmax(gate_logits, dim=1)
        elif self.gate_mode == 'sigmoid':
            weights = torch.sigmoid(gate_logits)
        else:
            raise ValueError(f"Unsupported gate_mode: {self.gate_mode}")
        
        # 加权
        w_vis = weights[:, 0:1]
        w_lidar = weights[:, 1:2]
        w_imu = weights[:, 2:3]
        
        fused_vis = vis * w_vis
        fused_lidar = lidar * w_lidar
        fused_imu = imu * w_imu
        
        # 再次拼接作为最终特征
        return torch.cat([fused_vis, fused_lidar, fused_imu], dim=1)

class CollisionModel(nn.Module):
    def __init__(self, activation='silu', dropout=0.5):
        super(CollisionModel, self).__init__()
        self.vis_encoder = VisionEncoder(activation=activation, dropout=dropout/2)
        self.lidar_encoder = LidarEncoder(activation=activation, dropout=dropout/2)
        self.imu_encoder = ImuEncoder(activation=activation, dropout=dropout/2)
        
        self.fusion = GatedFusion(gate_mode='softmax')
        
        # 碰撞预测头
        # 增加深度，缓解降维过快 (448 -> 256 -> 64 -> 1)
        self.head = nn.Sequential(
            nn.Linear(448, 256),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            _activation(activation),
            nn.Linear(64, 1)
        )
        
    def forward(self, image, lidar, imu):
        v = self.vis_encoder(image)
        l = self.lidar_encoder(lidar)
        i = self.imu_encoder(imu)
        
        fused = self.fusion(v, l, i)
        return self.head(fused)

class VOAModel(nn.Module):
    def __init__(self, action_output_mode='raw', action_v_max=0.3, action_w_max=1.5, activation='silu', dropout=0.5):
        super(VOAModel, self).__init__()
        self.vis_encoder = VisionEncoder(freeze_shallow=True, activation=activation, dropout=dropout/2)
        self.lidar_encoder = LidarEncoder(activation=activation, dropout=dropout/2)
        self.imu_encoder = ImuEncoder(activation=activation, dropout=dropout/2)
        
        self.fusion = GatedFusion(gate_mode='softmax')
        self.action_output_mode = action_output_mode
        self.action_v_max = float(action_v_max)
        self.action_w_max = float(action_w_max)
        
        # 1. Policy Head (用于正常行驶)
        # Output: [v, w]
        self.policy_head = nn.Sequential(
            nn.Linear(448, 256),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
        
        # 2. Recovery Head (用于 ANN 触发后的动作续接)
        # Output: [v_s, w_s, v_m, w_m, v_l, w_l] (6 dim)
        self.recovery_head = nn.Sequential(
            nn.Linear(448, 256),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(128, 6)
        )
        
    def forward(self, image, lidar, imu):
        v = self.vis_encoder(image)
        l = self.lidar_encoder(lidar)
        i = self.imu_encoder(imu)
        
        fused = self.fusion(v, l, i)
        
        policy_out = self.policy_head(fused)
        recovery_out = self.recovery_head(fused)

        if self.action_output_mode != 'raw':
            if self.action_output_mode == 'tanh_norm':
                policy_out = torch.tanh(policy_out)
                recovery_out = torch.tanh(recovery_out)
            elif self.action_output_mode == 'tanh_scaled':
                policy_scale = policy_out.new_tensor([self.action_v_max, self.action_w_max]).view(1, 2)
                recovery_scale = recovery_out.new_tensor([self.action_v_max, self.action_w_max] * 3).view(1, 6)
                policy_out = torch.tanh(policy_out) * policy_scale
                recovery_out = torch.tanh(recovery_out) * recovery_scale
            else:
                raise ValueError(f"Unsupported action_output_mode: {self.action_output_mode}")
        
        # 返回字典，方便 loss 计算
        return {
            'policy': policy_out,
            'recovery': recovery_out
        }
