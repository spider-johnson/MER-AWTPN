
# Adaptive Weighted Temporal Prototype Network (AWTPN)

Official PyTorch implementation for the paper:

> **Adaptive Weighted Temporal Prototype Network for Multimodal Emotion Recognition**  
> (Submitted to *Information Processing & Management*, 2025)

---

## 🧠 Model Overview
AWTPN consists of three components tightly matched to the paper:
- **Shared Emotional Space (SES):** lightweight linear projections mapping audio and video features into a unified space.
- **Temporal Prototypical Network (TPN):** a learnable prototype table; distances from each audio segment / video frame to category prototypes (Eq. 6–8) provide alignment signals.
- **Adaptive Weight Module (AWM):** frame/segment-wise attention that converts distances to reliability weights and fuses modalities (Eq. 15–21).

Losses implemented (Eq. 9–12, 23–25):
- Intra-/Inter-class metric losses, Align loss, Contrastive loss, plus CE and MSE; combined via λ/η weights.

---

## 🔧 Requirements

- Python 3.9+
- PyTorch 2.0+

```bash
pip install -r requirements.txt
```

---

## 💾 Datasets

We evaluate on three public datasets used in the paper:
- IEMOCAP
- MELD
- CMU-MOSEI

**We do not redistribute datasets.** See `data/README.md` for download instructions and expected folder layout.  
Preprocessing scripts are provided in `data/preprocess.py` and will cache processed tensors under `data/processed/`.

---

## 🚀 Quickstart

1) Edit a config file under `config/`, e.g. `config/iemocap.yaml`.  
2) Train:
```bash
python train.py --config config/iemocap.yaml
```
3) Evaluate:
```bash
python test.py --config config/iemocap.yaml --checkpoint checkpoints/awtpn_best.pth
```

> Minimal example uses dummy feature loaders if raw datasets are not present (to sanity-check the pipeline).

---

## 📊 Expected Results (from the paper)

| Dataset    | w-ACC | w-F1  |
|------------|------:|------:|
| IEMOCAP    | 77.21 | 78.25 |
| MELD       | 73.18 | 70.42 |
| CMU-MOSEI  | 61.09 | 61.29 |

(Exact numbers may vary with seeds, splits and encoder checkpoints.)

---

## 📁 Repository Structure

```
AWTPN/
├── README.md
├── requirements.txt
├── train.py
├── test.py
├── config/
│   ├── iemocap.yaml
│   ├── meld.yaml
│   └── mosei.yaml
├── models/
│   ├── __init__.py
│   ├── awtpn.py
│   ├── ses_module.py
│   ├── tpn_module.py
│   └── awm_module.py
├── utils/
│   ├── losses.py
│   ├── metrics.py
│   └── seed.py
├── data/
│   ├── README.md
│   └── preprocess.py
├── scripts/
│   ├── run_iemocap.sh
│   ├── run_meld.sh
│   └── run_mosei.sh
├── checkpoints/
│   └── README.md
├── results/
│   └── README.md
├── .gitignore
└── LICENSE
```

---

## 🔁 Reproducibility Notes

- We expose all λ/η loss weights and (α_A, β_V) fusion weights in YAML configs.  
- `utils/seed.py` sets seeds for NumPy/PyTorch and enables deterministic CuDNN (optional).  
- We log metrics per-epoch and save the best checkpoint by validation w-F1.

---

## 📘 Citation

```
@article{hao2025awtpn,
  title={Adaptive Weighted Temporal Prototype Network for Multimodal Emotion Recognition},
  author={Hao, Chenyu and collaborators},
  journal={Information Processing & Management},
  year={2025}
}
```

---

## 🪪 License

MIT License © 2025 Chenyu Hao
