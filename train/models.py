import torch
import torch.nn as nn
import torchvision.models as models

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
        x = x.mean([2, 3]) # [B, 1024]
        return x

class VisionEncoder(nn.Module):
    def __init__(self, output_dim=256, freeze_shallow=True):
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
            nn.Linear(1024, output_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        feat = self.backbone(x)
        return self.projector(feat)

class LidarEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super(LidarEncoder, self).__init__()
        # Input: [B, 1, 360]
        # Circular Padding 保持环形连续性
        self.net = nn.Sequential(
            # Conv1
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2, padding_mode='circular'),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            
            # Conv2
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            # Conv3
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            # Global Max Pooling
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [B, 360] -> [B, 1, 360]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.net(x)
        return self.fc(feat)

class ImuEncoder(nn.Module):
    def __init__(self, input_dim=9, output_dim=64):
        super(ImuEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # x: [B, 9]
        return self.net(x)

class GatedFusion(nn.Module):
    def __init__(self, vis_dim=256, lidar_dim=128, imu_dim=64):
        super(GatedFusion, self).__init__()
        self.total_dim = vis_dim + lidar_dim + imu_dim
        
        # 门控网络：根据拼接后的特征计算每个模态的权重
        self.gate_net = nn.Sequential(
            nn.Linear(self.total_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3), # 3个模态的权重
            nn.Sigmoid()
        )
        
        self.vis_dim = vis_dim
        self.lidar_dim = lidar_dim
        self.imu_dim = imu_dim

    def forward(self, vis, lidar, imu):
        # 拼接
        concat_feat = torch.cat([vis, lidar, imu], dim=1)
        
        # 计算权重 [B, 3]
        weights = self.gate_net(concat_feat)
        
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
    def __init__(self):
        super(CollisionModel, self).__init__()
        self.vis_encoder = VisionEncoder()
        self.lidar_encoder = LidarEncoder()
        self.imu_encoder = ImuEncoder()
        
        self.fusion = GatedFusion()
        
        # 碰撞预测头
        self.head = nn.Sequential(
            nn.Linear(448, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image, lidar, imu):
        v = self.vis_encoder(image)
        l = self.lidar_encoder(lidar)
        i = self.imu_encoder(imu)
        
        fused = self.fusion(v, l, i)
        prob = self.head(fused)
        return prob

class VOAModel(nn.Module):
    def __init__(self):
        super(VOAModel, self).__init__()
        self.vis_encoder = VisionEncoder(freeze_shallow=True)
        self.lidar_encoder = LidarEncoder()
        self.imu_encoder = ImuEncoder()
        
        self.fusion = GatedFusion()
        
        # 1. Policy Head (用于正常行驶)
        # Output: [v, w]
        self.policy_head = nn.Sequential(
            nn.Linear(448, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )
        
        # 2. Recovery Head (用于 ANN 触发后的动作续接)
        # Output: [v_s, w_s, v_m, w_m, v_l, w_l] (6 dim)
        self.recovery_head = nn.Sequential(
            nn.Linear(448, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)
        )
        
    def forward(self, image, lidar, imu):
        v = self.vis_encoder(image)
        l = self.lidar_encoder(lidar)
        i = self.imu_encoder(imu)
        
        fused = self.fusion(v, l, i)
        
        policy_out = self.policy_head(fused)
        recovery_out = self.recovery_head(fused)
        
        # 返回字典，方便 loss 计算
        return {
            'policy': policy_out,
            'recovery': recovery_out
        }
