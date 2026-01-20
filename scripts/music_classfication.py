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
- X_spec from scripts/extract_features.py: [n_mels x time_frames]
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow importing from project root when running this file directly.
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


def resolve_from_root(p: str | Path) -> Path:
	path = Path(p)
	return path if path.is_absolute() else (PROJECT_ROOT / path)


@dataclass
class TrainConfig:
	features_dir: Path
	labels_csv: Path
	run_dir: Path | None = None
	batch_size: int = 2
	epochs: int = 5
	lr: float = 1e-3
	dropout: float = 0.25
	kernel: int = 3
	target_frames: int = 1024
	split_ratio: float = 0.7
	num_workers: int = 0
	seed: int = 42
	model: str = "cnn"  # cnn | cnn_transformer
	tx_layers: int = 2
	tx_heads: int = 4
	tx_ff_dim: int = 256
	tx_dropout: float = 0.1


def _read_labels(labels_csv: Path) -> Dict[str, str]:
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
			f"You likely downloaded more WAVs and updated downloaded_authors.csv; now extract features for them. "
			f"Run: python scripts/extract_features.py --in-dir data --out-dir {features_dir.name} "
			f"--only-labeled-csv downloaded_authors.csv --skip-existing"
		)
	return items


def _make_label_vocab(items: List[Tuple[Path, str]]) -> Tuple[Dict[str, int], List[str]]:
	names = sorted({c for _, c in items}, key=lambda s: s.casefold())
	to_id = {name: i for i, name in enumerate(names)}
	return to_id, names


def build_model(
	*,
	num_classes: int,
	kernel: int,
	dropout: float,
	model_type: str,
	target_frames: int,
	debug_shapes: bool = False,
	tx_layers: int = 2,
	tx_heads: int = 4,
	tx_ff_dim: int = 256,
	tx_dropout: float = 0.1,
):
	import torch
	import torch.nn as nn

	if kernel not in (3, 5):
		raise ValueError("kernel must be 3 or 5")
	padding = kernel // 2

	class CNNFrontEnd(nn.Module):
		def __init__(self):
			super().__init__()

			def block(in_ch: int, out_ch: int) -> nn.Sequential:
				return nn.Sequential(
					nn.Conv2d(in_ch, out_ch, kernel_size=(kernel, kernel), stride=1, padding=padding),
					nn.BatchNorm2d(out_ch),
					nn.ReLU(inplace=True),
					nn.Dropout(p=dropout),
					# Pool only the frequency axis to preserve time resolution
					nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
				)

			self.net = nn.Sequential(
				block(1, 32),
				block(32, 64),
				block(64, 128),
			)

		def forward(self, x: torch.Tensor) -> torch.Tensor:
			# x: [B, 1, F, T] -> [B, C, F', T]
			return self.net(x)

	class CNNClassifier(nn.Module):
		def __init__(self):
			super().__init__()
			self.front = CNNFrontEnd()
			self.head = nn.Sequential(
				nn.AdaptiveAvgPool2d((1, 1)),
				nn.Flatten(),
				nn.Linear(128, num_classes),
			)

		def forward(self, x: torch.Tensor) -> torch.Tensor:
			x = self.front(x)
			return self.head(x)

	class SinusoidalPositionalEncoding(nn.Module):
		def __init__(self, d_model: int, max_len: int):
			super().__init__()
			pe = torch.zeros(max_len, d_model)
			position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
			div_term = torch.exp(
				torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model)
			)
			pe[:, 0::2] = torch.sin(position * div_term)
			pe[:, 1::2] = torch.cos(position * div_term)
			self.register_buffer("pe", pe, persistent=False)

		def forward(self, x: torch.Tensor) -> torch.Tensor:
			# x: [T, B, E]
			T = x.size(0)
			return x + self.pe[:T].unsqueeze(1)

	class CNNTransformerClassifier(nn.Module):
		def __init__(self):
			super().__init__()
			self.front = CNNFrontEnd()
			embed_dim = 128
			self._debug_shapes = bool(debug_shapes)
			self._debug_printed = False

			# Step 3: pool frequency axis -> sequence [T, B, E]
			self.pos = SinusoidalPositionalEncoding(d_model=embed_dim, max_len=max(1, int(target_frames)))

			# Step 4: Transformer encoder for global context
			enc_layer = nn.TransformerEncoderLayer(
				d_model=embed_dim,
				nhead=int(tx_heads),
				dim_feedforward=int(tx_ff_dim),
				dropout=float(tx_dropout),
				activation="relu",
				batch_first=False,
				norm_first=True,
			)
			self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(tx_layers))
			self.out = nn.Linear(embed_dim, num_classes)

		def forward(self, x: torch.Tensor) -> torch.Tensor:
			# x: [B, 1, F, T]
			z = self.front(x)  # [B, C, F', T]
			if self._debug_shapes and not self._debug_printed:
				print("F_cnn=", tuple(z.shape))
			# pool/flatten frequency axis -> [B, C, T]
			z = z.mean(dim=2)
			if self._debug_shapes and not self._debug_printed:
				print("F_seq_freqpooled=", tuple(z.shape))
			# transformer expects [T, B, E]
			z = z.permute(2, 0, 1)
			if self._debug_shapes and not self._debug_printed:
				print("F_seq_TBE=", tuple(z.shape))
			z = self.pos(z)
			z = self.encoder(z)
			if self._debug_shapes and not self._debug_printed:
				print("F_trans=", tuple(z.shape))
				self._debug_printed = True
			# aggregate time -> [B, E]
			z = z.mean(dim=0)
			return self.out(z)

	if model_type == "cnn":
		return CNNClassifier()
	if model_type == "cnn_transformer":
		return CNNTransformerClassifier()
	raise ValueError("model_type must be 'cnn' or 'cnn_transformer'")


def _split_indices(n: int, *, split_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
	if n < 2:
		raise ValueError("Need at least 2 samples to split")
	rng = np.random.default_rng(seed)
	indices = np.arange(n)
	rng.shuffle(indices)
	split_n = int(split_ratio * n)
	split_n = max(1, min(split_n, n - 1))
	return indices[:split_n], indices[split_n:]


def _default_ckpt_name(model_type: str) -> str:
	if model_type == "cnn_transformer":
		return "models/cnn_transformer_composer.pt"
	return "models/cnn_composer.pt"


def _extract_xspec_from_wav(wav_path: Path) -> np.ndarray:
	import librosa

	y, sr_actual = librosa.load(wav_path, sr=22050, mono=True)
	D = librosa.stft(y, n_fft=2048, hop_length=512)
	P = np.abs(D) ** 2
	mel_fb = librosa.filters.mel(sr=sr_actual, n_fft=2048, n_mels=128, fmin=0.0, fmax=8000.0)
	m = mel_fb @ P
	X_spec = librosa.power_to_db(m, ref=np.max)
	return X_spec.astype(np.float32)


def train(cfg: TrainConfig) -> int:
	import torch
	import torch.nn.functional as F
	from torch.utils.data import DataLoader, Dataset
	from scripts.run_logging import write_run_info_txt

	rng = np.random.default_rng(cfg.seed)
	labels = _read_labels(cfg.labels_csv)
	items = _build_dataset_index(cfg.features_dir, labels)
	vocab, id_to_name = _make_label_vocab(items)

	target_frames = int(cfg.target_frames)
	train_idx, val_idx = _split_indices(len(items), split_ratio=cfg.split_ratio, seed=cfg.seed)

	if cfg.run_dir is not None:
		write_run_info_txt(
			run_dir=cfg.run_dir,
			script_name="music_classfication",
			argv=sys.argv,
			args=cfg,
			extra={
				"num_items": len(items),
				"num_classes": len(id_to_name),
				"train_items": int(len(train_idx)),
				"val_items": int(len(val_idx)),
				"features_dir": str(cfg.features_dir),
				"labels_csv": str(cfg.labels_csv),
			},
		)

	class NpzDataset(Dataset):
		def __init__(self, idxs: np.ndarray):
			self.idxs = list(map(int, idxs))

		def __len__(self):
			return len(self.idxs)

		def __getitem__(self, i: int):
			npz_path, composer = items[self.idxs[i]]
			x = _load_npz_xspec(npz_path)
			x = _pad_or_crop_time(x, target_frames)
			x = torch.from_numpy(x[None, :, :])
			y = torch.tensor(vocab[composer], dtype=torch.long)
			return x, y

	train_loader = DataLoader(
		NpzDataset(train_idx), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
	)
	val_loader = DataLoader(
		NpzDataset(val_idx), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = build_model(
		num_classes=len(id_to_name),
		kernel=cfg.kernel,
		dropout=cfg.dropout,
		model_type=cfg.model,
		target_frames=target_frames,
		debug_shapes=False,
		tx_layers=cfg.tx_layers,
		tx_heads=cfg.tx_heads,
		tx_ff_dim=cfg.tx_ff_dim,
		tx_dropout=cfg.tx_dropout,
	).to(device)
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
			f"train_loss={train_loss / max(1, train_total):.4f} "
			f"train_acc={train_correct / max(1, train_total):.3f} "
			f"val_acc={val_correct / max(1, val_total):.3f}"
		)

	out_dir = PROJECT_ROOT / "models"
	out_dir.mkdir(parents=True, exist_ok=True)
	ckpt = out_dir / Path(_default_ckpt_name(cfg.model)).name
	torch.save(
		{
			"state_dict": model.state_dict(),
			"id_to_name": id_to_name,
			"kernel": cfg.kernel,
			"dropout": cfg.dropout,
			"model": cfg.model,
			"target_frames": target_frames,
			"split_ratio": cfg.split_ratio,
			"tx_layers": cfg.tx_layers,
			"tx_heads": cfg.tx_heads,
			"tx_ff_dim": cfg.tx_ff_dim,
			"tx_dropout": cfg.tx_dropout,
		},
		ckpt,
	)
	print(f"saved={ckpt}")
	return 0


def evaluate(
	*,
	ckpt_path: Path,
	features_dir: Path,
	labels_csv: Path,
	split_ratio: float,
	seed: int,
	batch_size: int,
	debug_shapes: bool,
) -> int:
	import torch
	from torch.utils.data import DataLoader, Dataset

	ckpt = torch.load(ckpt_path, map_location="cpu")
	id_to_name: list[str] = ckpt["id_to_name"]
	to_id = {name: i for i, name in enumerate(id_to_name)}

	labels = _read_labels(labels_csv)
	items_all = _build_dataset_index(features_dir, labels)
	items = [(p, c) for (p, c) in items_all if c in to_id]
	if len(items) < 2:
		raise ValueError("Not enough labeled items to evaluate")

	target_frames = int(ckpt.get("target_frames", 1024))
	val_train_idx, val_idx = _split_indices(len(items), split_ratio=split_ratio, seed=seed)
	_ = val_train_idx  # unused

	class NpzDataset(Dataset):
		def __init__(self, idxs: np.ndarray):
			self.idxs = list(map(int, idxs))

		def __len__(self):
			return len(self.idxs)

		def __getitem__(self, i: int):
			npz_path, composer = items[self.idxs[i]]
			x = _load_npz_xspec(npz_path)
			x = _pad_or_crop_time(x, target_frames)
			x = torch.from_numpy(x[None, :, :])
			y = torch.tensor(to_id[composer], dtype=torch.long)
			return x, y

	loader = DataLoader(NpzDataset(val_idx), batch_size=batch_size, shuffle=False)
	model = build_model(
		num_classes=len(id_to_name),
		kernel=int(ckpt.get("kernel", 3)),
		dropout=float(ckpt.get("dropout", 0.25)),
		model_type=str(ckpt.get("model", "cnn")),
		target_frames=target_frames,
		debug_shapes=debug_shapes,
		tx_layers=int(ckpt.get("tx_layers", 2)),
		tx_heads=int(ckpt.get("tx_heads", 4)),
		tx_ff_dim=int(ckpt.get("tx_ff_dim", 256)),
		tx_dropout=float(ckpt.get("tx_dropout", 0.1)),
	)
	model.load_state_dict(ckpt["state_dict"])
	model.eval()

	correct = 0
	total = 0
	with torch.no_grad():
		for xb, yb in loader:
			logits = model(xb)
			pred = logits.argmax(dim=1)
			correct += int((pred == yb).sum().item())
			total += int(xb.size(0))

	acc = correct / max(1, total)
	print(f"eval_samples={total}")
	print(f"eval_acc={acc:.3f}")
	return 0


def predict(*, ckpt_path: Path, wav_path: Path | None, npz_path: Path | None, topk: int = 3) -> int:
	import torch

	if (wav_path is None) == (npz_path is None):
		raise ValueError("Provide exactly one of wav_path or npz_path")
	ckpt = torch.load(ckpt_path, map_location="cpu")
	id_to_name = ckpt["id_to_name"]
	model = build_model(
		num_classes=len(id_to_name),
		kernel=int(ckpt.get("kernel", 3)),
		dropout=float(ckpt.get("dropout", 0.25)),
		model_type=str(ckpt.get("model", "cnn")),
		target_frames=int(ckpt.get("target_frames", 1024)),
		debug_shapes=False,
		tx_layers=int(ckpt.get("tx_layers", 2)),
		tx_heads=int(ckpt.get("tx_heads", 4)),
		tx_ff_dim=int(ckpt.get("tx_ff_dim", 256)),
		tx_dropout=float(ckpt.get("tx_dropout", 0.1)),
	)
	model.load_state_dict(ckpt["state_dict"])
	model.eval()

	if npz_path is not None:
		x = _load_npz_xspec(npz_path)
		source = npz_path.name
	else:
		x = _extract_xspec_from_wav(wav_path)
		source = wav_path.name

	x = _pad_or_crop_time(x, int(ckpt.get("target_frames", 1024)))
	xb = torch.from_numpy(x[None, None, :, :])
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
	p.add_argument("--mode", choices=["train", "predict", "eval"], default="train")
	p.add_argument("--features-dir", default="features")
	p.add_argument("--labels-csv", default="downloaded_authors.csv")
	p.add_argument("--run-dir", default="runs", help="Folder to write run_info.txt")
	p.add_argument("--run-name", default="", help="Optional tag to include in run folder name")
	p.add_argument("--epochs", type=int, default=5)
	p.add_argument("--batch-size", type=int, default=2)
	p.add_argument("--lr", type=float, default=1e-3)
	p.add_argument("--dropout", type=float, default=0.25)
	p.add_argument("--kernel", type=int, default=3, choices=[3, 5])
	p.add_argument("--target-frames", type=int, default=1024)
	p.add_argument("--split-ratio", type=float, default=0.7)
	p.add_argument("--seed", type=int, default=42)
	p.add_argument("--model", choices=["cnn", "cnn_transformer"], default="cnn")
	p.add_argument("--tx-layers", type=int, default=2)
	p.add_argument("--tx-heads", type=int, default=4)
	p.add_argument("--tx-ff-dim", type=int, default=256)
	p.add_argument("--tx-dropout", type=float, default=0.1)
	p.add_argument("--debug-shapes", action="store_true", help="Print CNN->sequence->Transformer tensor shapes once")
	p.add_argument("--ckpt", default="models/cnn_composer.pt")
	p.add_argument("--predict-wav", default=None)
	p.add_argument("--predict-npz", default=None)
	p.add_argument("--topk", type=int, default=3)
	args = p.parse_args()

	from scripts.run_logging import create_run_dir, write_run_info_txt
	base_run_dir = resolve_from_root(args.run_dir)
	run_dir = create_run_dir(
		base_dir=base_run_dir,
		script_name="music_classfication",
		run_name=str(args.run_name) if args.run_name else None,
	)

	# If user keeps the default ckpt path but chooses transformer, switch to transformer ckpt.
	ckpt_arg = args.ckpt
	if ckpt_arg == "models/cnn_composer.pt" and args.model == "cnn_transformer":
		ckpt_arg = _default_ckpt_name("cnn_transformer")
	ckpt_path = resolve_from_root(ckpt_arg)

	if args.mode == "predict":
		wav_path = resolve_from_root(args.predict_wav) if args.predict_wav else None
		npz_path = resolve_from_root(args.predict_npz) if args.predict_npz else None
		write_run_info_txt(
			run_dir=run_dir,
			script_name="music_classfication",
			argv=sys.argv,
			args=args,
			extra={
				"mode": "predict",
				"ckpt": str(ckpt_path),
				"predict_wav": str(wav_path) if wav_path else None,
				"predict_npz": str(npz_path) if npz_path else None,
			},
		)
		return predict(ckpt_path=ckpt_path, wav_path=wav_path, npz_path=npz_path, topk=args.topk)

	if args.mode == "eval":
		write_run_info_txt(
			run_dir=run_dir,
			script_name="music_classfication",
			argv=sys.argv,
			args=args,
			extra={
				"mode": "eval",
				"ckpt": str(ckpt_path),
				"features_dir": str(resolve_from_root(args.features_dir)),
				"labels_csv": str(resolve_from_root(args.labels_csv)),
			},
		)
		return evaluate(
			ckpt_path=ckpt_path,
			features_dir=resolve_from_root(args.features_dir),
			labels_csv=resolve_from_root(args.labels_csv),
			split_ratio=float(args.split_ratio),
			seed=int(args.seed),
			batch_size=int(args.batch_size),
			debug_shapes=bool(args.debug_shapes),
		)

	cfg = TrainConfig(
		features_dir=resolve_from_root(args.features_dir),
		labels_csv=resolve_from_root(args.labels_csv),
		run_dir=run_dir,
		epochs=args.epochs,
		batch_size=args.batch_size,
		lr=args.lr,
		dropout=args.dropout,
		kernel=args.kernel,
		target_frames=args.target_frames,
		split_ratio=args.split_ratio,
		model=args.model,
		tx_layers=args.tx_layers,
		tx_heads=args.tx_heads,
		tx_ff_dim=args.tx_ff_dim,
		tx_dropout=args.tx_dropout,
		seed=int(args.seed),
	)
	return train(cfg)


if __name__ == "__main__":
	raise SystemExit(main())

