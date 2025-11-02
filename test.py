
import argparse, yaml, os, torch
from torch.utils.data import DataLoader, TensorDataset
from utils.seed import set_seed
from utils.metrics import compute_metrics
from models import AWTPN

def load_split(path):
    obj = torch.load(path)
    return obj["audio_feats"], obj["video_feats"], obj["labels"]

def build_loader(split_dir, batch_size):
    a, v, y = load_split(os.path.join(split_dir, "test.pt"))
    ds = TensorDataset(a, v, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)

def main(cfg, ckpt):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"]=="cuda" else "cpu")
    set_seed(cfg["seed"])
    test_loader = build_loader(cfg["data_root"], 32)

    model = AWTPN(
        audio_dim=cfg["audio_dim"], video_dim=cfg["video_dim"],
        shared_dim=cfg["shared_dim"], num_classes=cfg["num_classes"],
        ses_dropout=cfg["model"]["ses_dropout"],
        distance_scale_audio=cfg["model"]["distance_scale_audio"],
        distance_scale_video=cfg["model"]["distance_scale_video"],
        fusion_alpha_audio=cfg["model"]["fusion_alpha_audio"],
        fusion_beta_video=cfg["model"]["fusion_beta_video"]
    ).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ys, yh = [], []
    with torch.no_grad():
        for a, v, y in test_loader:
            a, v = a.to(device), v.to(device)
            logits = model(a, v)["logits"]
            yh.extend(logits.argmax(dim=-1).cpu().tolist())
            ys.extend(y.cpu().tolist())

    m = compute_metrics(ys, yh)
    print("Test metrics:", m)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.checkpoint)
