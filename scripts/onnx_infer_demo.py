import argparse
import ast
import os

import cv2
import numpy as np
import pandas as pd


def _load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image = cv2.resize(image, (160, 120))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    return image


def _parse_list_cell(cell, expected_len):
    if isinstance(cell, str):
        values = ast.literal_eval(cell)
    else:
        values = cell
    arr = np.asarray(values, dtype=np.float32)
    if expected_len is not None and arr.size != expected_len:
        raise ValueError(f"Expected length {expected_len}, got {arr.size}")
    return arr


def _get_dt_seconds(prev_row, row, default_dt=0.1):
    for col in ("timestamp_sec", "timestamp"):
        if col in row and pd.notna(row[col]) and col in prev_row and pd.notna(prev_row[col]):
            if col == "timestamp":
                return max((float(row[col]) - float(prev_row[col])) / 1e9, 0.0)
            return max(float(row[col]) - float(prev_row[col]), 0.0)
    return default_dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default="data_ready_2/data.csv")
    parser.add_argument("--root_dir", type=str, default="data_ready_2")
    parser.add_argument("--ann_onnx", type=str, default="checkpoints/ann_model.onnx")
    parser.add_argument("--voa_onnx", type=str, default="checkpoints/voa_model.onnx")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--risk_stop", type=float, default=0.5)
    parser.add_argument("--risk_resume", type=float, default=0.3)
    parser.add_argument("--hard_stop_dist", type=float, default=0.22)
    parser.add_argument("--recovery_frames", type=int, default=5)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError("onnxruntime is required to run this demo") from e

    ann_path = os.path.abspath(args.ann_onnx)
    voa_path = os.path.abspath(args.voa_onnx)
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"ANN ONNX not found: {ann_path}")
    if not os.path.exists(voa_path):
        raise FileNotFoundError(f"VOA ONNX not found: {voa_path}")

    ann_sess = ort.InferenceSession(ann_path, providers=["CPUExecutionProvider"])
    voa_sess = ort.InferenceSession(voa_path, providers=["CPUExecutionProvider"])

    data_csv = os.path.abspath(args.data_csv)
    root_dir = os.path.abspath(args.root_dir)
    df = pd.read_csv(data_csv)

    braking = False
    brake_elapsed = 0.0
    recovery_frames_left = 0

    prev_row = None

    for i in range(args.start, min(args.start + args.steps, len(df))):
        row = df.iloc[i]
        if prev_row is None:
            prev_row = row
        dt = _get_dt_seconds(prev_row, row)

        image_rel = row["image_path"]
        image_abs = os.path.join(root_dir, image_rel)
        image = _load_image(image_abs)[None, ...]
        lidar = (_parse_list_cell(row["lidar_ranges"], 360) / 8.0)[None, ...]
        imu = _parse_list_cell(row["imu_data"], 9)[None, ...]

        ort_inputs = {"image": image, "lidar": lidar, "imu": imu}
        risk = float(ann_sess.run(None, ort_inputs)[0].reshape(-1)[0])
        policy, recovery = voa_sess.run(None, ort_inputs)

        policy_v, policy_w = float(policy.reshape(-1)[0]), float(policy.reshape(-1)[1])
        rec = recovery.reshape(-1).tolist()
        v_s, w_s, v_m, w_m, v_l, w_l = [float(x) for x in rec[:6]]

        min_lidar = float(np.min(lidar))
        hard_stop = min_lidar < args.hard_stop_dist

        if risk >= args.risk_stop or hard_stop:
            braking = True
            brake_elapsed += dt
            recovery_frames_left = args.recovery_frames
            cmd_v, cmd_w = 0.0, 0.0
        else:
            if braking and risk <= args.risk_resume:
                if recovery_frames_left > 0:
                    recovery_frames_left -= 1
                    if brake_elapsed < 0.3:
                        cmd_v, cmd_w = v_m, w_m
                    else:
                        cmd_v, cmd_w = v_l, w_l
                else:
                    braking = False
                    brake_elapsed = 0.0
                    cmd_v, cmd_w = policy_v, policy_w
            else:
                cmd_v, cmd_w = policy_v, policy_w

        print(
            f"step={i} risk={risk:.3f} min_lidar={min_lidar:.3f} "
            f"braking={int(braking)} brake_t={brake_elapsed:.2f} cmd=({cmd_v:.3f},{cmd_w:.3f})"
        )

        prev_row = row


if __name__ == "__main__":
    main()

