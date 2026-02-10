import torch

from models import CollisionModel, VOAModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2

    image = torch.randn(batch_size, 3, 120, 160, device=device)
    lidar = torch.randn(batch_size, 360, device=device)
    imu = torch.randn(batch_size, 9, device=device)

    ann = CollisionModel().to(device).eval()
    voa = VOAModel(action_output_mode="tanh_scaled", action_v_max=0.3, action_w_max=1.5).to(device).eval()

    with torch.no_grad():
        risk_logits = ann(image, lidar, imu)
        assert list(risk_logits.shape) == [batch_size, 1], f"Unexpected ANN shape: {risk_logits.shape}"

        out = voa(image, lidar, imu)
        policy = out["policy"]
        recovery = out["recovery"]
        assert list(policy.shape) == [batch_size, 2], f"Unexpected policy shape: {policy.shape}"
        assert list(recovery.shape) == [batch_size, 6], f"Unexpected recovery shape: {recovery.shape}"

        v_max = 0.3
        w_max = 1.5
        assert float(policy[:, 0].abs().max().cpu()) <= v_max + 1e-4
        assert float(policy[:, 1].abs().max().cpu()) <= w_max + 1e-4
        assert float(recovery[:, 0::2].abs().max().cpu()) <= v_max + 1e-4
        assert float(recovery[:, 1::2].abs().max().cpu()) <= w_max + 1e-4

    print("OK")


if __name__ == "__main__":
    main()

