
import argparse, yaml, os, torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils.seed import set_seed
from utils.losses import total_loss
from utils.metrics import compute_metrics
from models import AWTPN

def load_split(path):
    obj = torch.load(path)
    return obj["audio_feats"], obj["video_feats"], obj["labels"]

def build_loader(split_dir, split, batch_size, shuffle=False):
    a, v, y = load_split(os.path.join(split_dir, f"{split}.pt"))
    ds = TensorDataset(a, v, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"]=="cuda" else "cpu")
    set_seed(cfg["seed"])

    train_loader = build_loader(cfg["data_root"], "train", cfg["train"]["batch_size"], shuffle=True)
    val_loader   = build_loader(cfg["data_root"], "val",   cfg["train"]["batch_size"], shuffle=False)

    model = AWTPN(
        audio_dim=cfg["audio_dim"], video_dim=cfg["video_dim"],
        shared_dim=cfg["shared_dim"], num_classes=cfg["num_classes"],
        ses_dropout=cfg["model"]["ses_dropout"],
        distance_scale_audio=cfg["model"]["distance_scale_audio"],
        distance_scale_video=cfg["model"]["distance_scale_video"],
        fusion_alpha_audio=cfg["model"]["fusion_alpha_audio"],
        fusion_beta_video=cfg["model"]["fusion_beta_video"]
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    best_f1 = -1.0
    os.makedirs(cfg["save_dir"], exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"]+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for step, (a, v, y) in enumerate(pbar, start=1):
            a, v, y = a.to(device), v.to(device), y.to(device)
            out = model(a, v)
            loss, parts = total_loss(out, y, cfg["loss"], cfg["num_classes"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        # validation
        model.eval()
        ys, yh = [], []
        with torch.no_grad():
            for a, v, y in val_loader:
                a, v = a.to(device), v.to(device)
                logits = model(a, v)["logits"]
                yh.extend(logits.argmax(dim=-1).cpu().tolist())
                ys.extend(y.cpu().tolist())
        m = compute_metrics(ys, yh)
        print("Val metrics:", {k: (v if k!='confusion_matrix' else 'cm') for k,v in m.items()})
        if m["w-F1"] > best_f1:
            best_f1 = m["w-F1"]
            torch.save(model.state_dict(), os.path.join(cfg["save_dir"], "awtpn_best.pth"))
            print(f"Saved best checkpoint (w-F1={best_f1:.4f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
