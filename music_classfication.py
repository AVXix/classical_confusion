"""Music classification (composer) using log-mel spectrograms.

Model architecture matches the spec:
- Conv2d kernels: 3x3 or 5x5 (configurable)
- Filters per layer: 32 -> 64 -> 128
- Stride: 1
- Pooling: only frequency axis (preserve temporal resolution)
- Activation: ReLU
- BatchNorm: Yes
- Dropout: 0.2-0.3

Input per clip:
- X_spec from extract_features.py: [n_mels x time_frames]

This is a minimal, runnable baseline. With only ~10 clips, this is mainly for
verifying the pipeline end-to-end.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class TrainConfig:
	features_dir: Path
	labels_csv: Path
	batch_size: int = 2
	epochs: int = 5
	lr: float = 1e-3
	dropout: float = 0.25
	kernel: int = 3  # 3 or 5
	target_frames: int = 1024
	split_ratio: float = 0.7
	num_workers: int = 0
	seed: int = 42


def _read_labels(labels_csv: Path) -> Dict[str, str]:
	"""Return mapping: feature_stem -> canonical_composer.

We join using the downloaded wav basename without extension.
Example:
  downloaded_file = "...Track05_wav.wav" -> stem "...Track05_wav"
  feature file = "...Track05_wav.npz" -> stem "...Track05_wav"
"""
	mapping: Dict[str, str] = {}
	with labels_csv.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		if not reader.fieldnames:
			raise ValueError("labels CSV has no headers")
		required = ["downloaded_file", "canonical_composer"]
		for r in required:
			if r not in reader.fieldnames:
				raise ValueError(f"labels CSV missing column: {r}")
		for row in reader:
			wav_name = (row.get("downloaded_file") or "").strip()
			composer = (row.get("canonical_composer") or "").strip()
			if not wav_name or not composer:
				continue
			stem = Path(wav_name).stem
			mapping[stem] = composer
	if not mapping:
		raise ValueError("No labels found in labels CSV")
	return mapping


def _pad_or_crop_time(x: np.ndarray, target_frames: int) -> np.ndarray:
	"""Pad/crop time axis to fixed length for batching."""
	if x.shape[1] == target_frames:
		return x
	if x.shape[1] > target_frames:
		return x[:, :target_frames]
	pad = target_frames - x.shape[1]
	return np.pad(x, ((0, 0), (0, pad)), mode="constant")


def _load_npz_xspec(npz_path: Path) -> np.ndarray:
	with np.load(npz_path) as d:
		if "X_spec" not in d:
			raise ValueError(f"Missing X_spec in {npz_path}")
		x = d["X_spec"].astype(np.float32)
	if x.ndim != 2:
		raise ValueError(f"X_spec must be 2D [n_mels x frames], got {x.shape}")
	return x


def _build_dataset_index(features_dir: Path, labels: Dict[str, str]) -> List[Tuple[Path, str]]:
	items: List[Tuple[Path, str]] = []
	for npz_path in sorted(features_dir.glob("*.npz")):
		stem = npz_path.stem
		if stem in labels:
			items.append((npz_path, labels[stem]))
	if not items:
		raise ValueError(
			f"No matching feature files in {features_dir}. "
			f"Expected stems: {sorted(labels.keys())[:5]}..."
		)
	return items


def _make_label_vocab(items: List[Tuple[Path, str]]) -> Tuple[Dict[str, int], List[str]]:
	names = sorted({c for _, c in items}, key=lambda s: s.casefold())
	to_id = {name: i for i, name in enumerate(names)}
	return to_id, names


def build_model(num_classes: int, kernel: int, dropout: float):
	import torch
	import torch.nn as nn

	if kernel not in (3, 5):
		raise ValueError("kernel must be 3 or 5")
	padding = kernel // 2

	def block(in_ch: int, out_ch: int) -> nn.Sequential:
		return nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=(kernel, kernel), stride=1, padding=padding),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Dropout(p=dropout),
			# Pool ONLY in frequency axis: (freq, time)
			nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
		)

	model = nn.Sequential(
		block(1, 32),
		block(32, 64),
		block(64, 128),
		# After convs: [B, 128, F', T]. Reduce to [B, 128] via mean over F and T.
		nn.AdaptiveAvgPool2d((1, 1)),
		nn.Flatten(),
		nn.Linear(128, num_classes),
	)
	return model


def _extract_xspec_from_wav(
	wav_path: Path,
	*,
	sr: int = 22050,
	n_fft: int = 2048,
	hop_length: int = 512,
	n_mels: int = 128,
	fmin: float = 0.0,
	fmax: float | None = 8000.0,
	log_mode: str = "db",
) -> np.ndarray:
	import librosa

	y, sr_actual = librosa.load(wav_path, sr=sr, mono=True)
	D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
	P = np.abs(D) ** 2
	mel_fb = librosa.filters.mel(
		sr=sr_actual,
		n_fft=n_fft,
		n_mels=n_mels,
		fmin=fmin,
		fmax=fmax,
	)
	mel_power = mel_fb @ P
	if log_mode == "db":
		X_spec = librosa.power_to_db(mel_power, ref=np.max)
	elif log_mode == "log":
		X_spec = np.log(mel_power + 1e-10)
	else:
		raise ValueError("log_mode must be 'db' or 'log'")
	return X_spec.astype(np.float32)


def train(cfg: TrainConfig) -> int:
	import torch
	from torch.utils.data import Dataset, DataLoader
	import torch.nn.functional as F

	rng = np.random.default_rng(cfg.seed)

	labels = _read_labels(cfg.labels_csv)
	items = _build_dataset_index(cfg.features_dir, labels)
	vocab, id_to_name = _make_label_vocab(items)

	# Use a fixed frame length for batching. Default is intentionally small to keep
	# training fast on CPU.
	target_frames = int(cfg.target_frames)
	if target_frames <= 0:
		raise ValueError("target_frames must be > 0")

	# Tiny dataset split (default 70/30)
	indices = np.arange(len(items))
	rng.shuffle(indices)
	split_n = int(cfg.split_ratio * len(items))
	split_n = max(1, min(split_n, len(items) - 1))
	train_idx = indices[:split_n]
	val_idx = indices[split_n:]

	class NpzDataset(Dataset):
		def __init__(self, idxs: np.ndarray):
			self.idxs = list(map(int, idxs))

		def __len__(self):
			return len(self.idxs)

		def __getitem__(self, i: int):
			npz_path, composer = items[self.idxs[i]]
			x = _load_npz_xspec(npz_path)
			x = _pad_or_crop_time(x, target_frames)
			# [1, n_mels, frames] for Conv2d
			x = torch.from_numpy(x[None, :, :])
			y = torch.tensor(vocab[composer], dtype=torch.long)
			return x, y

	train_loader = DataLoader(NpzDataset(train_idx), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
	val_loader = DataLoader(NpzDataset(val_idx), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = build_model(num_classes=len(id_to_name), kernel=cfg.kernel, dropout=cfg.dropout).to(device)
	opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

	print(f"classes={len(id_to_name)}")
	print("class_names=" + ", ".join(id_to_name))
	print(f"target_frames={target_frames}")
	print(f"device={device}")

	for epoch in range(1, cfg.epochs + 1):
		model.train()
		train_loss = 0.0
		train_correct = 0
		train_total = 0
		for xb, yb in train_loader:
			xb = xb.to(device)
			yb = yb.to(device)
			logits = model(xb)
			loss = F.cross_entropy(logits, yb)
			opt.zero_grad(set_to_none=True)
			loss.backward()
			opt.step()
			train_loss += float(loss.item()) * xb.size(0)
			train_correct += int((logits.argmax(dim=1) == yb).sum().item())
			train_total += int(xb.size(0))

		model.eval()
		val_correct = 0
		val_total = 0
		with torch.no_grad():
			for xb, yb in val_loader:
				xb = xb.to(device)
				yb = yb.to(device)
				logits = model(xb)
				val_correct += int((logits.argmax(dim=1) == yb).sum().item())
				val_total += int(xb.size(0))

		print(
			f"epoch={epoch} "
			f"train_loss={train_loss/max(1,train_total):.4f} "
			f"train_acc={train_correct/max(1,train_total):.3f} "
			f"val_acc={val_correct/max(1,val_total):.3f}"
		)

	# Save checkpoint + label vocab
	out_dir = cfg.features_dir.parent / "models"
	out_dir.mkdir(parents=True, exist_ok=True)
	ckpt = out_dir / "cnn_composer.pt"
	torch.save(
		{
			"state_dict": model.state_dict(),
			"id_to_name": id_to_name,
			"kernel": cfg.kernel,
			"dropout": cfg.dropout,
			"target_frames": target_frames,
			"split_ratio": cfg.split_ratio,
		},
		ckpt,
	)
	print(f"saved={ckpt}")
	return 0


def predict(
	*,
	ckpt_path: Path,
	wav_path: Path | None,
	npz_path: Path | None,
	topk: int = 3,
) -> int:
	import torch

	if (wav_path is None) == (npz_path is None):
		raise ValueError("Provide exactly one of wav_path or npz_path")

	if not ckpt_path.exists():
		print(f"Missing checkpoint: {ckpt_path}")
		return 2

	ckpt = torch.load(ckpt_path, map_location="cpu")
	id_to_name = ckpt["id_to_name"]
	kernel = int(ckpt.get("kernel", 3))
	dropout = float(ckpt.get("dropout", 0.25))
	target_frames = int(ckpt.get("target_frames", 1024))

	model = build_model(num_classes=len(id_to_name), kernel=kernel, dropout=dropout)
	model.load_state_dict(ckpt["state_dict"])
	model.eval()

	if npz_path is not None:
		x = _load_npz_xspec(npz_path)
		source = npz_path.name
	else:
		x = _extract_xspec_from_wav(wav_path)
		source = wav_path.name

	x = _pad_or_crop_time(x, target_frames)
	xb = torch.from_numpy(x[None, None, :, :])  # [1, 1, n_mels, frames]

	with torch.no_grad():
		logits = model(xb)
		probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

	topk = max(1, min(int(topk), len(id_to_name)))
	idx = np.argsort(-probs)[:topk]

	print(f"source={source}")
	print(f"predicted={id_to_name[int(idx[0])]}")
	print("topk=")
	for i in idx:
		print(f"  {id_to_name[int(i)]}: {float(probs[int(i)]):.4f}")

	return 0


def main() -> int:
	p = argparse.ArgumentParser()
	p.add_argument("--mode", choices=["train", "predict"], default="train")
	p.add_argument("--features-dir", default="features", help="Folder containing .npz feature files")
	p.add_argument("--labels-csv", default="downloaded_authors.csv", help="CSV mapping downloaded_file -> canonical_composer")
	p.add_argument("--epochs", type=int, default=5)
	p.add_argument("--batch-size", type=int, default=2)
	p.add_argument("--lr", type=float, default=1e-3)
	p.add_argument("--dropout", type=float, default=0.25)
	p.add_argument("--kernel", type=int, default=3, choices=[3, 5])
	p.add_argument("--target-frames", type=int, default=1024, help="Time frames to pad/crop X_spec for batching")
	p.add_argument("--split-ratio", type=float, default=0.7, help="Train/val split ratio (e.g., 0.7 for 70/30)")
	p.add_argument("--ckpt", default="models/cnn_composer.pt", help="Checkpoint path for predict")
	p.add_argument("--predict-wav", default=None, help="Path to a .wav file to predict")
	p.add_argument("--predict-npz", default=None, help="Path to a .npz feature file to predict")
	p.add_argument("--topk", type=int, default=3)
	args = p.parse_args()

	if args.mode == "predict":
		wav_path = Path(args.predict_wav) if args.predict_wav else None
		npz_path = Path(args.predict_npz) if args.predict_npz else None
		return predict(
			ckpt_path=Path(args.ckpt),
			wav_path=wav_path,
			npz_path=npz_path,
			topk=args.topk,
		)

	cfg = TrainConfig(
		features_dir=Path(args.features_dir),
		labels_csv=Path(args.labels_csv),
		epochs=args.epochs,
		batch_size=args.batch_size,
		lr=args.lr,
		dropout=args.dropout,
		kernel=args.kernel,
		target_frames=args.target_frames,
		split_ratio=args.split_ratio,
	)
	return train(cfg)


if __name__ == "__main__":
	raise SystemExit(main())

