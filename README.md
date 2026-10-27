# RNEPNet

## Setup

Linux x86_64, Python 3.10, PyTorch 2.4.1 / CUDA 12.1.

```bash
conda create -n rnepnet python=3.10 -y
conda activate rnepnet
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Weights

Download the complete RNEPNet weights from Baidu Netdisk and extract to:

```text
weights/us3d.pth
weights/whu.pth
weights/diachronic.pth
```

## Run

```bash
bash scripts/evaluate_us3d.sh /path/to/US3D
bash scripts/evaluate_whu.sh /path/to/WHU
bash scripts/evaluate_diachronic.sh /path/to/DiachronicStereo

# Cross-dataset evaluation
bash scripts/evaluate_us3d_to_whu.sh /path/to/WHU
bash scripts/evaluate_whu_to_us3d.sh /path/to/US3D
```

[Dataset layouts](docs/datasets.md)

Choose weights and the target dataset directly:

```bash
python evaluate.py --checkpoint weights/us3d.pth --dataset whu --data-root /path/to/WHU
```
