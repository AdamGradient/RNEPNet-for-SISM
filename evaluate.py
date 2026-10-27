#!/usr/bin/env python3
"""Evaluate RNEPNet checkpoints or save disparity predictions."""
import argparse
import json
import math
import shlex
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def read_image(path, dataset):
    path = Path(path)
    if path.suffix.lower() in (".iio", ".npy"):
        image = np.load(path)
    elif path.suffix.lower() in (".tif", ".tiff"):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if dataset == "whu" and image.dtype == np.uint16:
            image = image.astype(np.float32)
            span = image.max() - image.min()
            image = ((image - image.min()) / span * 255).astype(np.uint8)
    else:
        image = np.asarray(Image.open(path))
    image = np.asarray(image).astype(np.uint8)
    if image.ndim == 2:
        image = np.tile(image[..., None], (1, 1, 3))
    return torch.from_numpy(np.ascontiguousarray(image[..., :3].transpose(2, 0, 1))).float()


def read_gt(path):
    path = Path(path)
    if path.suffix.lower() in (".iio", ".npy"):
        disparity = np.load(path)
    else:
        disparity = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if disparity.ndim == 3 and disparity.shape[-1] == 1:
        disparity = disparity[..., 0]
    return torch.from_numpy(np.ascontiguousarray(disparity, dtype=np.float32))


def native_pairs(root, dataset):
    if dataset == "diachronic":
        aois = set((root / "test_aois.csv").read_text().splitlines())
        left = [p for p in sorted((root / "L").glob("*.iio")) if p.name[:7] in aois]
        return [(p, root / "R" / p.name, root / "disparity" / p.name) for p in left]
    split = root / "test"
    columns = [[], [], []]
    if dataset == "us3d":
        patterns = ("RGB/*LEFT_RGB.tif", "RGB/*RIGHT_RGB.tif", "Truth/*DSP.tif")
        columns = [sorted(split.glob(pattern)) for pattern in patterns]
    else:
        for city in ("HY", "KM", "QC", "SG", "WH", "YD"):
            patterns = (f"left/{city}_*.tiff", f"right/{city}*.tiff", f"disp/{city}*.tiff")
            city_columns = [sorted(split.glob(pattern)) for pattern in patterns]
            for output, values in zip(columns, city_columns):
                output.extend(values)
    return list(zip(*columns))


def manifest_pairs(path, root):
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = shlex.split(line, comments=True)
        if not fields:
            continue
        paths = [Path(value) if Path(value).is_absolute() else root / value for value in fields]
        pairs.append(tuple(paths + [None] * (3 - len(paths))))
    return pairs


class Pairs(Dataset):
    def __init__(self, rows, dataset):
        self.rows, self.dataset = rows, dataset

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        left, right, gt = self.rows[index]
        a, b = read_image(left, self.dataset), read_image(right, self.dataset)
        disparity = read_gt(gt) if gt is not None else torch.empty(0)
        return index, a, b, disparity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("us3d", "whu", "diachronic"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--pairs", type=Path, help="Text file: left right [gt]; paths relative to data-root")
    parser.add_argument("--output-dir", type=Path, help="Save original-resolution float32 disparity .npy files")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable BF16 automatic mixed precision (default); --no-amp uses FP32")
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, help="Evaluate only the first N pairs (partial result)")
    args = parser.parse_args()
    rows = manifest_pairs(args.pairs, args.data_root) if args.pairs else native_pairs(args.data_root, args.dataset)
    if args.max_samples is not None:
        rows = rows[:args.max_samples]
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    import rnep_runtime
    torch.set_num_threads(4)
    engine = rnep_runtime.load(str(args.checkpoint), args.dataset, device=args.device,
                               amp=args.amp, iterations=args.iterations)
    loader = DataLoader(Pairs(rows, args.dataset), batch_size=1, shuffle=False,
                        num_workers=4, pin_memory=args.device.startswith("cuda"))
    absolute = squared = 0.0
    bad = valid_pixels = ground_truth_images = 0
    lower, upper = engine.dataset_range
    with torch.inference_mode():
        for index, left, right, gt in loader:
            index = int(index.item())
            prediction = engine.predict(left, right).float()
            if args.output_dir is not None:
                name = f"{index:05d}_{rows[index][0].stem}.npy"
                np.save(args.output_dir / name, prediction.squeeze().cpu().numpy().astype(np.float32))
            if gt.numel():
                ground_truth_images += 1
                gt = gt.to(prediction.device, non_blocking=True).reshape_as(prediction)
                mask = (gt > lower) & (gt < upper)
                error = (prediction - gt).abs()[mask]
                bad += int((error > 3).sum().item())
                error = error.double()
                valid_pixels += error.numel()
                absolute += error.sum().item()
                squared += error.square().sum().item()
            if (index + 1) % 50 == 0 or index + 1 == len(rows):
                print(json.dumps({"completed": index + 1, "total": len(rows)}), flush=True)
    result = {"dataset": args.dataset, "source_dataset": engine.source_dataset,
              "images": len(rows), "ground_truth_images": ground_truth_images,
              "valid_pixels": valid_pixels, "partial": args.max_samples is not None,
              "amp": args.amp, "precision": "bf16" if args.amp else "fp32", "iterations": args.iterations,
              "epe": absolute / valid_pixels if valid_pixels else None,
              "d1": 100.0 * bad / valid_pixels if valid_pixels else None,
              "rmse": math.sqrt(squared / valid_pixels) if valid_pixels else None}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
