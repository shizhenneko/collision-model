import os
import csv
import ast
import argparse
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

# Optional image backend
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class CollisionDataset(Dataset):
    """
    Dataset that loads image, lidar, imu, action [v,w], collision label.
    Expects CSV fields: timestamp, image_path (optional), lidar_ranges, imu_data (optional), linear_x, angular_z, label_medium (optional), episode_id (optional).
    """
    def __init__(self, root_dir: str, max_time_gap: float = 1.0, img_w: int = 160, img_h: int = 120, horizons: List[float] = None):
        self.root_dir = root_dir
        self.csv_path = os.path.join(root_dir, 'data.csv')
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Data CSV not found at {self.csv_path}")
        # Load rows
        self.rows = []
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append(r)
        # Parse timestamp
        def parse_float(x, default=None):
            try:
                return float(x)
            except Exception:
                return default
        self.timestamp = [parse_float(r.get('timestamp', None), None) for r in self.rows]
        # Build episode_id from CSV or timestamp segmentation
        if 'episode_id' in (self.rows[0].keys() if self.rows else []):
            try:
                self.episode_id = [int(float(r.get('episode_id', 0))) for r in self.rows]
            except Exception:
                self.episode_id = self._segment_by_time(self.timestamp, max_time_gap)
        else:
            self.episode_id = self._segment_by_time(self.timestamp, max_time_gap)
        # Cache image paths
        self.image_paths = [r.get('image_path', '') for r in self.rows]
        # Targets
        self.vx = [parse_float(r.get('linear_x', 0.0), 0.0) for r in self.rows]
        self.wz = [parse_float(r.get('angular_z', 0.0), 0.0) for r in self.rows]
        # Collision label may be missing, default 0
        self.collision = []
        for r in self.rows:
            val = r.get('label_medium', None)
            try:
                self.collision.append(float(val))
            except Exception:
                self.collision.append(0.0)
        self.img_w = img_w
        self.img_h = img_h
        # horizons for multi-timehead targets
        self.horizons = sorted(horizons) if horizons else []
        # precompute future targets per horizon using timestamp alignment within episodes
        self._compute_future_targets()

    def _segment_by_time(self, ts: List[float], max_gap: float) -> List[int]:
        eps = []
        current = 0
        prev = None
        for t in ts:
            if prev is None or t is None or prev is None:
                eps.append(current)
            else:
                if t - prev > max_gap:
                    current += 1
                eps.append(current)
            prev = t
        return eps

    def __len__(self):
        return len(self.rows)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        if not rel_path:
            return torch.zeros(3, self.img_h, self.img_w, dtype=torch.float32)
        full = os.path.join(self.root_dir, rel_path)
        if HAS_CV2:
            img = cv2.imread(full)
            if img is None:
                return torch.zeros(3, self.img_h, self.img_w, dtype=torch.float32)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            return img
        else:
            return torch.zeros(3, self.img_h, self.img_w, dtype=torch.float32)

    def _parse_list(self, s, dtype=float) -> np.ndarray:
        try:
            arr = np.array(ast.literal_eval(s), dtype=dtype)
            return arr
        except Exception:
            return np.zeros(0, dtype=dtype)

    def _compute_future_targets(self):
        # Initialize dicts for future targets
        self.vx_h = {h: [0.0] * len(self.rows) for h in self.horizons}
        self.wz_h = {h: [0.0] * len(self.rows) for h in self.horizons}
        self.collision_h = {h: [0.0] * len(self.rows) for h in self.horizons}
        N = len(self.rows)
        for i in range(N):
            ts_i = self.timestamp[i]
            ep_i = self.episode_id[i]
            for h in self.horizons:
                j = None
                if ts_i is not None:
                    for k in range(i + 1, N):
                        if self.episode_id[k] != ep_i:
                            break
                        ts_k = self.timestamp[k]
                        if ts_k is None:
                            continue
                        if ts_k - ts_i >= h:
                            j = k
                            break
                if j is None:
                    # Fallback to current targets if no future point found in the same episode
                    self.vx_h[h][i] = self.vx[i]
                    self.wz_h[h][i] = self.wz[i]
                    self.collision_h[h][i] = self.collision[i]
                else:
                    self.vx_h[h][i] = self.vx[j]
                    self.wz_h[h][i] = self.wz[j]
                    self.collision_h[h][i] = self.collision[j]

    def __getitem__(self, idx):
        row = self.rows[idx]
        # Image
        img = self._load_image(row.get('image_path', ''))
        # Lidar
        lidar = self._parse_list(row.get('lidar_ranges', '[]'), dtype=np.float32)
        if lidar.size == 0:
            lidar = np.zeros(360, dtype=np.float32)
        lidar = np.clip(lidar, 0.05, 8.0)
        lidar_t = torch.from_numpy(lidar).float()  # shape [360]
        # IMU
        imu = self._parse_list(row.get('imu_data', '[]'), dtype=np.float32)
        if imu.size == 0:
            imu = np.zeros(6, dtype=np.float32)
        imu_t = torch.from_numpy(imu).float()
        # Targets (current)
        action = torch.tensor([self.vx[idx], self.wz[idx]], dtype=torch.float32)
        collision = torch.tensor([self.collision[idx]], dtype=torch.float32)
        sample = {
            'image': img,
            'lidar': lidar_t,
            'imu': imu_t,
            'action': action,
            'collision': collision,
            'episode_id': self.episode_id[idx]
        }
        # Add multi-horizon targets if configured
        for h in self.horizons:
            sample[f'action_h_{h}'] = torch.tensor([self.vx_h[h][idx], self.wz_h[h][idx]], dtype=torch.float32)
            sample[f'collision_h_{h}'] = torch.tensor([self.collision_h[h][idx]], dtype=torch.float32)
        return sample


def split_concat_by_episode(dsets: List[CollisionDataset], val_ratio: float = 0.2) -> Tuple[Subset, Subset]:
    """Split ConcatDataset by episode across multiple folders, aiming for val_ratio samples."""
    # Build episode counts across datasets
    ep_entries = []  # list of (ds_idx, episode_id, count)
    total_samples = 0
    for di, ds in enumerate(dsets):
        # count per episode within dataset
        counts = {}
        for i, ep in enumerate(ds.episode_id):
            counts[ep] = counts.get(ep, 0) + 1
        for ep, cnt in counts.items():
            ep_entries.append((di, ep, cnt))
            total_samples += cnt
    # shuffle episodes
    random.seed(42)
    random.shuffle(ep_entries)
    target_val = int(total_samples * val_ratio)
    current_val = 0
    val_episodes = {di: set() for di in range(len(dsets))}
    for di, ep, cnt in ep_entries:
        if current_val < target_val:
            val_episodes[di].add(ep)
            current_val += cnt
        else:
            break
    # Build global indices for ConcatDataset
    train_indices = []
    val_indices = []
    # global offset
    offsets = []
    off = 0
    for ds in dsets:
        offsets.append(off)
        off += len(ds)
    for di, ds in enumerate(dsets):
        val_eps = val_episodes[di]
        for i, ep in enumerate(ds.episode_id):
            gi = offsets[di] + i
            if ep in val_eps:
                val_indices.append(gi)
            else:
                train_indices.append(gi)
    concat = ConcatDataset(dsets)
    return Subset(concat, train_indices), Subset(concat, val_indices)


class MultiModalNet(nn.Module):
    def __init__(self, action_head=True, collision_head=True, horizons: List[float] = None):
        super().__init__()
        self.horizons = horizons or []
        # Image branch
        self.img = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        # Lidar branch (1D conv)
        self.lidar = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        # IMU branch
        self.imu = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU()
        )
        # Fusion
        self.fuse = nn.Sequential(
            nn.Linear(64 + 64 + 32, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.use_action = action_head
        self.use_collision = collision_head
        if action_head:
            self.head_action = nn.Linear(64, 2)
        if collision_head:
            self.head_collision = nn.Linear(64, 1)
        # dynamic multi-horizon heads
        # sanitize horizon keys for ModuleDict (no dots allowed in module names)
        def _san_hkey(h: float) -> str:
            return f"h_{str(h).replace('.', '_')}"
        if action_head and self.horizons:
            self.head_action_h = nn.ModuleDict({ _san_hkey(h): nn.Linear(64, 2) for h in self.horizons })
        else:
            self.head_action_h = None
        if collision_head and self.horizons:
            self.head_collision_h = nn.ModuleDict({ _san_hkey(h): nn.Linear(64, 1) for h in self.horizons })
        else:
            self.head_collision_h = None

    def forward(self, image, lidar, imu):
        # image: [B,3,120,160], lidar: [B,360], imu: [B,6]
        img_feat = self.img(image).view(image.size(0), -1)
        lid_in = lidar.unsqueeze(1)
        lid_feat = self.lidar(lid_in).view(image.size(0), -1)
        imu_feat = self.imu(imu)
        z = self.fuse(torch.cat([img_feat, lid_feat, imu_feat], dim=1))
        outs = {}
        if self.use_action:
            outs['action'] = self.head_action(z)
        if self.use_collision:
            outs['collision'] = self.head_collision(z)
        # sanitize horizon keys in forward
        def _san_hkey(h: float) -> str:
            return f"h_{str(h).replace('.', '_')}"
        if self.head_action_h is not None:
            for h in self.horizons:
                outs[f'action_h_{h}'] = self.head_action_h[_san_hkey(h)](z)
        if self.head_collision_h is not None:
            for h in self.horizons:
                outs[f'collision_h_{h}'] = self.head_collision_h[_san_hkey(h)](z)
        return outs


def train_supervised(args):
    # build datasets from subfolders
    subdirs = [d for d in sorted(os.listdir(args.input_parent_dir)) if os.path.isdir(os.path.join(args.input_parent_dir, d))]
    dsets = []
    for name in subdirs:
        p = os.path.join(args.input_parent_dir, name)
        if os.path.exists(os.path.join(p, 'data.csv')):
            try:
                dsets.append(CollisionDataset(p, max_time_gap=args.max_time_gap, horizons=args.horizons))
                print(f"Loaded {len(dsets[-1])} samples from {name}")
            except Exception as e:
                print(f"[Skip] {name}: {e}")
    if len(dsets) == 0:
        print("No datasets found under input_parent_dir")
        return
    # split
    train_set, val_set = split_concat_by_episode(dsets, val_ratio=args.val_ratio)
    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # model
    model = MultiModalNet(action_head=args.predict_action, collision_head=args.predict_collision, horizons=args.horizons)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    # train loop
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = []
        for batch in train_loader:
            img = batch['image'].to(device).float()
            lidar = batch['lidar'].to(device).float()
            imu = batch['imu'].to(device).float()
            outs = model(img, lidar, imu)
            # reward/importance weights
            if args.reward_weighting == 'by_collision':
                w = (1.0 + args.reward_lambda * batch['collision'].to(device).float().squeeze(1))
            elif args.reward_weighting == 'by_lidar_min':
                dmin = torch.clamp(lidar.min(dim=1).values, min=1e-2)
                w = 1.0 + args.reward_lambda * (1.0 / dmin)
            else:
                w = torch.ones(img.size(0), device=device)
            loss = 0.0
            if args.predict_action:
                tgt_a = batch['action'].to(device).float()
                loss_a_vec = ((outs['action'] - tgt_a) ** 2).mean(dim=1)
                loss += (loss_a_vec * w).mean()
                # multi-horizon action heads
                for h in args.horizons:
                    key_a = f'action_h_{h}'
                    if key_a in outs:
                        tgt_ah = batch[key_a].to(device).float()
                        loss_ah_vec = ((outs[key_a] - tgt_ah) ** 2).mean(dim=1)
                        loss += (loss_ah_vec * w).mean()
            if args.predict_collision:
                tgt_c = batch['collision'].to(device).float()
                loss_c_vec = bce(outs['collision'], tgt_c).squeeze(1)
                loss += (loss_c_vec * w).mean()
                # multi-horizon collision heads
                for h in args.horizons:
                    key_c = f'collision_h_{h}'
                    if key_c in outs:
                        tgt_ch = batch[key_c].to(device).float()
                        loss_ch_vec = bce(outs[key_c], tgt_ch).squeeze(1)
                        loss += (loss_ch_vec * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss.append(loss.item())
        # val
        model.eval()
        val_action_mse = []
        val_collision_bce = []
        val_collision_acc = []
        val_action_mse_h = {h: [] for h in args.horizons}
        val_collision_bce_h = {h: [] for h in args.horizons}
        val_collision_acc_h = {h: [] for h in args.horizons}
        with torch.no_grad():
            for batch in val_loader:
                img = batch['image'].to(device).float()
                lidar = batch['lidar'].to(device).float()
                imu = batch['imu'].to(device).float()
                outs = model(img, lidar, imu)
                if args.predict_action:
                    tgt_a = batch['action'].to(device).float()
                    val_action_mse.append(F.mse_loss(outs['action'], tgt_a).item())
                    for h in args.horizons:
                        key_a = f'action_h_{h}'
                        if key_a in outs:
                            tgt_ah = batch[key_a].to(device).float()
                            val_action_mse_h[h].append(F.mse_loss(outs[key_a], tgt_ah).item())
                if args.predict_collision:
                    tgt_c = batch['collision'].to(device).float()
                    loss_c = bce(outs['collision'], tgt_c).mean().item()
                    val_collision_bce.append(loss_c)
                    pred = (torch.sigmoid(outs['collision']) > 0.5).float()
                    acc = (pred == tgt_c).float().mean().item()
                    val_collision_acc.append(acc)
                    for h in args.horizons:
                        key_c = f'collision_h_{h}'
                        if key_c in outs:
                            tgt_ch = batch[key_c].to(device).float()
                            loss_ch = bce(outs[key_c], tgt_ch).mean().item()
                            val_collision_bce_h[h].append(loss_ch)
                            pred_h = (torch.sigmoid(outs[key_c]) > 0.5).float()
                            acc_h = (pred_h == tgt_ch).float().mean().item()
                            val_collision_acc_h[h].append(acc_h)
        print(f"Epoch {ep}/{args.epochs} | train_loss={np.mean(tr_loss):.4f} "
              f"val_mse={np.mean(val_action_mse) if val_action_mse else 0:.4f} "
              f"val_bce={np.mean(val_collision_bce) if val_collision_bce else 0:.4f} "
              f"val_acc={np.mean(val_collision_acc) if val_collision_acc else 0:.4f}")
    # save model
    ensure_dir(args.output_dir)
    out_path = os.path.join(args.output_dir, 'supervised_multimodal.pt')
    torch.save(model.state_dict(), out_path)
    print(f"Model saved to {out_path}")
    # save val metrics CSV
    metrics_path = os.path.join(args.output_dir, 'supervised_val_metrics.csv')
    try:
        with open(metrics_path, 'w', newline='') as f:
            wcsv = csv.writer(f)
            # header
            header = ['val_action_mse', 'val_collision_bce', 'val_collision_acc']
            for h in args.horizons:
                header += [f'val_action_mse_h_{h}', f'val_collision_bce_h_{h}', f'val_collision_acc_h_{h}']
            wcsv.writerow(header)
            # values
            row = [
                f"{np.mean(val_action_mse) if val_action_mse else 0:.6f}",
                f"{np.mean(val_collision_bce) if val_collision_bce else 0:.6f}",
                f"{np.mean(val_collision_acc) if val_collision_acc else 0:.6f}"
            ]
            for h in args.horizons:
                row += [
                    f"{np.mean(val_action_mse_h[h]) if val_action_mse_h[h] else 0:.6f}",
                    f"{np.mean(val_collision_bce_h[h]) if val_collision_bce_h[h] else 0:.6f}",
                    f"{np.mean(val_collision_acc_h[h]) if val_collision_acc_h[h] else 0:.6f}"
                ]
            wcsv.writerow(row)
        print(f"Validation metrics saved to {metrics_path}")
    except Exception as e:
        print(f"Failed to save validation metrics: {e}")


class ExportWrapper(nn.Module):
    def __init__(self, net: nn.Module, horizons: List[float]):
        super().__init__()
        self.net = net
        self.horizons = horizons or []
    def forward(self, image, lidar, imu):
        outs = self.net(image, lidar, imu)
        tensors = []
        # action heads
        if 'action' in outs:
            tensors.append(outs['action'])
        for h in self.horizons:
            key_a = f'action_h_{h}'
            if key_a in outs:
                tensors.append(outs[key_a])
        # collision heads (optional)
        if 'collision' in outs:
            tensors.append(outs['collision'])
        for h in self.horizons:
            key_c = f'collision_h_{h}'
            if key_c in outs:
                tensors.append(outs[key_c])
        return tuple(tensors)


def export_to_onnx(args):
    # Rebuild model and load weights on CPU
    model = MultiModalNet(action_head=args.predict_action, collision_head=args.predict_collision, horizons=args.horizons)
    model.load_state_dict(torch.load(args.pt_path, map_location='cpu'))
    model.eval(); model.to('cpu')
    wrapper = ExportWrapper(model, args.horizons)
    # Dummy inputs on CPU
    dummy_image = torch.zeros(1, 3, 120, 160, dtype=torch.float32)
    dummy_lidar = torch.zeros(1, 360, dtype=torch.float32)
    dummy_imu = torch.zeros(1, 6, dtype=torch.float32)
    # Names for inputs/outputs
    input_names = ['image', 'lidar', 'imu']
    output_names = []
    if args.predict_action:
        output_names.append('action')
        for h in args.horizons:
            output_names.append(f'action_h_{h}')
    if args.predict_collision:
        output_names.append('collision')
        for h in args.horizons:
            output_names.append(f'collision_h_{h}')
    dynamic_axes = {
        'image': {0: 'batch'},
        'lidar': {0: 'batch'},
        'imu': {0: 'batch'}
    }
    for name in output_names:
        dynamic_axes[name] = {0: 'batch'}
    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_lidar, dummy_imu),
        args.export_onnx,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    print(f"ONNX exported to {args.export_onnx}")


def parse_args():
    p = argparse.ArgumentParser(description='Supervised multi-modal training with CollisionDataset & ConcatDataset')
    p.add_argument('--input_parent_dir', type=str, required=True, help='Parent dir containing multiple cleaned data folders')
    p.add_argument('--output_dir', type=str, required=True, help='Output dir to save model and metrics')
    p.add_argument('--max_time_gap', type=float, default=1.0, help='Episode segmentation gap by timestamp (seconds)')
    p.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio by episode counts')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--predict_action', action='store_true', help='Enable action [v,w] regression head')
    p.add_argument('--predict_collision', action='store_true', help='Enable collision classification head')
    # new args for multi-horizon and reward-weighted regression
    p.add_argument('--horizons', type=str, default='0.1,0.5,1.0', help='Comma-separated future horizons in seconds, e.g., "0.1,0.5,1.0"')
    p.add_argument('--reward_weighting', type=str, choices=['none', 'by_collision', 'by_lidar_min'], default='none', help='Weight losses by collision label or inverse lidar min distance')
    p.add_argument('--reward_lambda', type=float, default=1.0, help='Strength of reward/importance weighting')
    p.add_argument('--export_onnx', type=str, default=None, help='If set, export ONNX to this path and exit')
    p.add_argument('--pt_path', type=str, default=None, help='Path to .pt state_dict to load for ONNX export')
    args = p.parse_args()
    # parse horizons string to list of floats
    args.horizons = [float(x.strip()) for x in args.horizons.split(',') if x.strip()]
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.export_onnx:
        if args.pt_path is None:
            args.pt_path = os.path.join(args.output_dir, 'supervised_multimodal.pt')
        export_to_onnx(args)
    else:
        train_supervised(args)