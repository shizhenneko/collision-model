import torch
import torch.onnx
import torch.nn as nn
import os
from models import VOAModel, CollisionModel

class VOAExportWrapper(nn.Module):
    def __init__(self, voa_model):
        super().__init__()
        self.voa_model = voa_model

    def forward(self, image, lidar, imu):
        outputs = self.voa_model(image, lidar, imu)
        return outputs['policy'], outputs['recovery']

class ANNExportWrapper(nn.Module):
    def __init__(self, ann_model):
        super().__init__()
        self.ann_model = ann_model

    def forward(self, image, lidar, imu):
        return torch.sigmoid(self.ann_model(image, lidar, imu))

def export_model(model_class, checkpoint_path, output_path, is_dual_head=False, model_kwargs=None, wrap_single_head_sigmoid=False):
    print(f"Exporting {checkpoint_path} to {output_path}...")
    
    # 1. 初始化模型
    model_kwargs = model_kwargs or {}
    model = model_class(**model_kwargs)
        
    # 2. 加载权重
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            print("  Checkpoint loaded successfully.")
        except Exception as e:
            print(f"  Error loading checkpoint: {e}")
            return
    else:
        print(f"  Warning: Checkpoint {checkpoint_path} not found. Exporting random weights for testing.")
        
    model.eval()
    model_to_export = model
    if is_dual_head:
        model_to_export = VOAExportWrapper(model)
        model_to_export.eval()
    elif wrap_single_head_sigmoid:
        model_to_export = ANNExportWrapper(model)
        model_to_export.eval()
    
    # 3. 创建 Dummy Input
    # Image: [1, 3, 120, 160]
    dummy_image = torch.randn(1, 3, 120, 160)
    # Lidar: [1, 360]
    dummy_lidar = torch.randn(1, 360)
    # Imu: [1, 9]
    dummy_imu = torch.randn(1, 9)
    
    # 4. 导出 ONNX
    output_names = ['policy', 'recovery'] if is_dual_head else ['risk']
    dynamic_axes = {
        'image': {0: 'batch_size'},
        'lidar': {0: 'batch_size'},
        'imu': {0: 'batch_size'}
    }
    for name in output_names:
        dynamic_axes[name] = {0: 'batch_size'}

    torch.onnx.export(
        model_to_export,
        (dummy_image, dummy_lidar, dummy_imu),
        output_path,
        export_params=True,        # 存储权重
        opset_version=18,          # 使用 Opset 18，避免 PyTorch 2.x 的降级转换错误
        do_constant_folding=True,  # 优化常量
        input_names=['image', 'lidar', 'imu'],
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    print(f"  Export success: {output_path}")
    
    # 5. 验证 ONNX 模型 (Optional)
    try:
        import onnxruntime as ort
        import numpy as np
        
        ort_session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        
        ort_inputs = {
            'image': dummy_image.numpy(),
            'lidar': dummy_lidar.numpy(),
            'imu': dummy_imu.numpy()
        }
        
        ort_outs = ort_session.run(None, ort_inputs)
        print("  ONNX Runtime verification passed.")
        for i, out in enumerate(ort_outs):
            print(f"  Output {i} shape: {out.shape}")
        
    except ImportError:
        print("  onnxruntime not installed. Skipping verification.")
    except Exception as e:
        print(f"  ONNX verification failed: {e}")

def main():
    save_dir = '../checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    ACTION_V_MAX = 0.3
    ACTION_W_MAX = 1.5
    
    # 导出 VOA 模型 (双头)
    export_model(
        VOAModel,
        os.path.join(save_dir, 'voa_best.pth'), 
        os.path.join(save_dir, 'voa_model.onnx'),
        is_dual_head=True,
        model_kwargs={
            'action_output_mode': 'tanh_scaled',
            'action_v_max': ACTION_V_MAX,
            'action_w_max': ACTION_W_MAX
        }
    )
    
    # 导出 ANN 模型 (单头)
    export_model(
        CollisionModel, 
        os.path.join(save_dir, 'ann_best.pth'), 
        os.path.join(save_dir, 'ann_model.onnx'),
        is_dual_head=False,
        wrap_single_head_sigmoid=True
    )

if __name__ == '__main__':
    main()
