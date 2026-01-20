#!/usr/bin/env python3
"""VAE over CNN pooled embeddings + fusion classifier.

Pipeline:
1) Load CNN/Transformer from checkpoint to extract pooled embeddings.
2) Train VAE on CNN pooled embeddings (input x -> mu, logvar -> recon x_hat).
3) Build fused vector: [cnn_pooled | transformer_pooled | vae_z | handcrafted].
4) Train a softmax (cross-entropy) classifier on fused vectors.

Inputs:
- features/*.npz created by scripts/extract_features.py (X_spec, optional X_hand)
- downloaded_authors.csv for labels (use --label-column for composer_era)
- models/cnn_transformer_composer.pt (or cnn) for embedding extractor

Outputs:
- models/vae_fusion_classifier.pt
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_logging import create_run_dir, write_run_info_txt
from scripts import music_classfication as mc


def resolve_from_root(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _read_labels(labels_csv: Path, label_column: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("labels CSV has no headers")
        required = ["downloaded_file", label_column]
        for r in required:
            if r not in reader.fieldnames:
                raise ValueError(f"labels CSV missing column: {r}")
        for row in reader:
            wav_name = (row.get("downloaded_file") or "").strip()
            label = (row.get(label_column) or "").strip()
            if not wav_name or not label:
                continue
            stem = Path(wav_name).stem
            mapping[stem] = label
    if not mapping:
        raise ValueError("No labels found in labels CSV")
    return mapping


def _build_dataset_index(features_dir: Path, labels: Dict[str, str]) -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []
    for npz_path in sorted(features_dir.glob("*.npz")):
        stem = npz_path.stem
        if stem in labels:
            items.append((npz_path, labels[stem]))
    if not items:
        raise ValueError(f"No matching feature files in {features_dir}.")
    return items


def _make_label_vocab(items: List[Tuple[Path, str]]) -> Tuple[Dict[str, int], List[str]]:
    names = sorted({c for _, c in items}, key=lambda s: s.casefold())
    to_id = {name: i for i, name in enumerate(names)}
    return to_id, names


def _pad_or_crop_time(x: np.ndarray, target_frames: int) -> np.ndarray:
    if x.shape[1] == target_frames:
        return x
    if x.shape[1] > target_frames:
        return x[:, :target_frames]
    pad = target_frames - x.shape[1]
    return np.pad(x, ((0, 0), (0, pad)), mode="constant")


def _load_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(npz_path) as d:
        if "X_spec" not in d:
            raise ValueError(f"Missing X_spec in {npz_path}")
        x_spec = d["X_spec"].astype(np.float32)
        x_hand = d["X_hand"].astype(np.float32) if "X_hand" in d else None
    if x_spec.ndim != 2:
        raise ValueError(f"X_spec must be 2D [n_mels x frames], got {x_spec.shape}")
    if x_hand is not None and x_hand.ndim != 1:
        raise ValueError(f"X_hand must be 1D, got {x_hand.shape}")
    return x_spec, x_hand


def _extract_xspec_from_wav(wav_path: Path) -> np.ndarray:
    import librosa

    y, sr_actual = librosa.load(wav_path, sr=22050, mono=True)
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    P = np.abs(D) ** 2
    mel_fb = librosa.filters.mel(sr=sr_actual, n_fft=2048, n_mels=128, fmin=0.0, fmax=8000.0)
    m = mel_fb @ P
    X_spec = librosa.power_to_db(m, ref=np.max)
    return X_spec.astype(np.float32)


def _split_indices(n: int, *, split_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("Need at least 2 samples to split")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    split_n = int(split_ratio * n)
    split_n = max(1, min(split_n, n - 1))
    return idx[:split_n], idx[split_n:]


@dataclass
class FeatureExtractorConfig:
    ckpt_path: Path
    target_frames: int
    device: torch.device


class TorchFeatureExtractor:
    def __init__(self, cfg: FeatureExtractorConfig):
        self._torch = torch
        self.ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
        self.target_frames = int(cfg.target_frames)
        self.device = cfg.device

        self.model = mc.build_model(
            num_classes=len(self.ckpt["id_to_name"]),
            kernel=int(self.ckpt.get("kernel", 3)),
            dropout=float(self.ckpt.get("dropout", 0.25)),
            model_type=str(self.ckpt.get("model", "cnn")),
            target_frames=int(self.ckpt.get("target_frames", self.target_frames)),
            debug_shapes=False,
            tx_layers=int(self.ckpt.get("tx_layers", 2)),
            tx_heads=int(self.ckpt.get("tx_heads", 4)),
            tx_ff_dim=int(self.ckpt.get("tx_ff_dim", 256)),
            tx_dropout=float(self.ckpt.get("tx_dropout", 0.1)),
        ).to(self.device)
        self.model.load_state_dict(self.ckpt["state_dict"], strict=True)
        self.model.eval()

        self.front = getattr(self.model, "front", None)
        if self.front is None:
            raise ValueError("Expected model to have attribute 'front' for CNN feature extraction")

        self.is_transformer = str(self.ckpt.get("model", "cnn")) == "cnn_transformer"
        self.encoder = getattr(self.model, "encoder", None)
        self.pos = getattr(self.model, "pos", None)

    def embed(self, x_spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_spec = _pad_or_crop_time(x_spec, int(self.ckpt.get("target_frames", self.target_frames)))
        xb = torch.from_numpy(x_spec[None, None, :, :]).to(self.device)

        with torch.no_grad():
            z = self.front(xb)
            z_seq = z.mean(dim=2)
            cnn_pooled = z_seq.mean(dim=2).squeeze(0).cpu().numpy()

            if self.is_transformer and self.encoder is not None and self.pos is not None:
                tbe = z_seq.permute(2, 0, 1)
                tbe = self.pos(tbe)
                tbe = self.encoder(tbe)
                trans_pooled = tbe.mean(dim=0).squeeze(0).cpu().numpy()
            else:
                trans_pooled = cnn_pooled.copy()

        return cnn_pooled.astype(np.float32), trans_pooled.astype(np.float32)


class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.enc_fc1 = nn.Linear(input_dim, 512)
        self.enc_fc2 = nn.Linear(512, 256)
        self.enc_mu = nn.Linear(256, latent_dim)
        self.enc_logvar = nn.Linear(256, latent_dim)

        self.dec_fc1 = nn.Linear(latent_dim, 256)
        self.dec_fc2 = nn.Linear(256, 512)
        self.dec_out = nn.Linear(512, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        return self.dec_out(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class FusionClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


def _standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std_safe = np.where(std == 0, 1.0, std)
    return (x - mean) / std_safe


def _get_device(device: str) -> torch.device:
    requested = str(device or "auto").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_pipeline(
    *,
    features_dir: Path,
    labels_csv: Path,
    label_column: str,
    ckpt_path: Path,
    target_frames: int,
    split_ratio: float,
    seed: int,
    require_handcrafted: bool,
    latent_dim: int,
    beta: float,
    vae_epochs: int,
    clf_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    out_path: Path,
    run_dir: Path | None,
) -> int:
    labels = _read_labels(labels_csv, label_column)
    items = _build_dataset_index(features_dir, labels)
    to_id, id_to_name = _make_label_vocab(items)

    extractor = TorchFeatureExtractor(FeatureExtractorConfig(ckpt_path=ckpt_path, target_frames=target_frames, device=device))

    cnn_list: list[np.ndarray] = []
    tx_list: list[np.ndarray] = []
    hand_list: list[np.ndarray] = []
    y_list: list[int] = []

    for npz_path, label in items:
        x_spec, x_hand = _load_npz(npz_path)
        cnn_p, tx_p = extractor.embed(x_spec)
        if x_hand is None:
            if require_handcrafted:
                raise ValueError("X_hand is missing. Re-run feature extraction with --handcrafted.")
            x_hand = np.zeros((47,), dtype=np.float32)
        cnn_list.append(cnn_p)
        tx_list.append(tx_p)
        hand_list.append(x_hand.astype(np.float32))
        y_list.append(int(to_id[label]))

    cnn = np.stack(cnn_list, axis=0)
    tx = np.stack(tx_list, axis=0)
    hand = np.stack(hand_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)

    train_idx, val_idx = _split_indices(len(items), split_ratio=split_ratio, seed=seed)

    # Normalize CNN features for VAE
    cnn_mean = cnn[train_idx].mean(axis=0)
    cnn_std = cnn[train_idx].std(axis=0)
    cnn_norm = _standardize(cnn, cnn_mean, cnn_std)

    vae = VAE(input_dim=cnn_norm.shape[1], latent_dim=int(latent_dim)).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(1, int(vae_epochs) + 1):
        vae.train()
        perm = np.random.permutation(train_idx)
        total_loss = 0.0
        for i in range(0, len(perm), batch_size):
            batch_idx = perm[i : i + batch_size]
            xb = torch.from_numpy(cnn_norm[batch_idx]).to(device)
            recon, mu, logvar = vae(xb)
            recon_loss = F.mse_loss(recon, xb, reduction="mean")
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + float(beta) * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
        avg_loss = total_loss / max(1, len(train_idx))
        print(f"vae_epoch={epoch} loss={avg_loss:.4f}")

    vae.eval()
    with torch.no_grad():
        mu_all = []
        for i in range(0, len(cnn_norm), batch_size):
            xb = torch.from_numpy(cnn_norm[i : i + batch_size]).to(device)
            mu, _ = vae.encode(xb)
            mu_all.append(mu.cpu().numpy())
        z = np.concatenate(mu_all, axis=0)

    fused = np.concatenate([cnn, tx, z, hand], axis=1).astype(np.float32)
    fused_mean = fused[train_idx].mean(axis=0)
    fused_std = fused[train_idx].std(axis=0)
    fused_norm = _standardize(fused, fused_mean, fused_std)

    clf = FusionClassifier(input_dim=fused_norm.shape[1], num_classes=len(id_to_name)).to(device)
    clf_opt = torch.optim.Adam(clf.parameters(), lr=lr)

    for epoch in range(1, int(clf_epochs) + 1):
        clf.train()
        perm = np.random.permutation(train_idx)
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for i in range(0, len(perm), batch_size):
            batch_idx = perm[i : i + batch_size]
            xb = torch.from_numpy(fused_norm[batch_idx]).to(device)
            yb = torch.from_numpy(y[batch_idx]).to(device)
            logits = clf(xb)
            loss = F.cross_entropy(logits, yb)
            clf_opt.zero_grad(set_to_none=True)
            loss.backward()
            clf_opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total_count += int(xb.size(0))

        clf.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i in range(0, len(val_idx), batch_size):
                batch_idx = val_idx[i : i + batch_size]
                xb = torch.from_numpy(fused_norm[batch_idx]).to(device)
                yb = torch.from_numpy(y[batch_idx]).to(device)
                logits = clf(xb)
                val_correct += int((logits.argmax(dim=1) == yb).sum().item())
                val_total += int(xb.size(0))

        print(
            f"clf_epoch={epoch} loss={total_loss / max(1, total_count):.4f} "
            f"train_acc={total_correct / max(1, total_count):.3f} "
            f"val_acc={val_correct / max(1, val_total):.3f}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vae_state": vae.state_dict(),
            "clf_state": clf.state_dict(),
            "id_to_name": id_to_name,
            "label_column": label_column,
            "cnn_mean": cnn_mean,
            "cnn_std": cnn_std,
            "fused_mean": fused_mean,
            "fused_std": fused_std,
            "latent_dim": int(latent_dim),
            "beta": float(beta),
            "ckpt_path": str(ckpt_path),
            "target_frames": int(target_frames),
            "require_handcrafted": bool(require_handcrafted),
        },
        out_path,
    )
    print(f"saved={out_path}")

    if run_dir is not None:
        write_run_info_txt(
            run_dir=run_dir,
            script_name="vae_fusion_classifier",
            argv=sys.argv,
            args={},
            extra={
                "features_dir": str(features_dir),
                "labels_csv": str(labels_csv),
                "label_column": label_column,
                "embed_ckpt": str(ckpt_path),
                "latent_dim": int(latent_dim),
                "beta": float(beta),
                "vae_epochs": int(vae_epochs),
                "clf_epochs": int(clf_epochs),
                "batch_size": int(batch_size),
                "lr": float(lr),
                "num_items": len(items),
                "num_classes": len(id_to_name),
            },
        )

    return 0


def predict(
    *,
    model_path: Path,
    ckpt_path: Path,
    wav_path: Path | None,
    npz_path: Path | None,
    target_frames: int,
    topk: int,
    device: torch.device,
) -> int:
    if (wav_path is None) == (npz_path is None):
        raise ValueError("Provide exactly one of --predict-wav or --predict-npz")

    bundle = torch.load(model_path, map_location="cpu")
    id_to_name: list[str] = bundle["id_to_name"]
    latent_dim = int(bundle["latent_dim"])
    require_handcrafted = bool(bundle.get("require_handcrafted", False))

    extractor = TorchFeatureExtractor(
        FeatureExtractorConfig(ckpt_path=ckpt_path, target_frames=target_frames, device=device)
    )

    if npz_path is not None:
        x_spec, x_hand = _load_npz(npz_path)
        source = npz_path.name
    else:
        x_spec = _extract_xspec_from_wav(wav_path)
        x_hand = None
        source = wav_path.name

    cnn_p, tx_p = extractor.embed(x_spec)
    if x_hand is None:
        if require_handcrafted:
            raise ValueError("X_hand is missing. Re-run feature extraction with --handcrafted.")
        x_hand = np.zeros((47,), dtype=np.float32)

    cnn_mean = bundle["cnn_mean"]
    cnn_std = bundle["cnn_std"]
    fused_mean = bundle["fused_mean"]
    fused_std = bundle["fused_std"]

    cnn_norm = _standardize(cnn_p[None, :], cnn_mean, cnn_std)

    vae = VAE(input_dim=int(cnn_norm.shape[1]), latent_dim=latent_dim).to(device)
    vae.load_state_dict(bundle["vae_state"], strict=True)
    vae.eval()

    with torch.no_grad():
        mu, _ = vae.encode(torch.from_numpy(cnn_norm).to(device))
        z = mu.cpu().numpy()

    fused = np.concatenate([cnn_p[None, :], tx_p[None, :], z, x_hand[None, :]], axis=1)
    fused_norm = _standardize(fused, fused_mean, fused_std)

    clf = FusionClassifier(input_dim=fused_norm.shape[1], num_classes=len(id_to_name)).to(device)
    clf.load_state_dict(bundle["clf_state"], strict=True)
    clf.eval()

    with torch.no_grad():
        logits = clf(torch.from_numpy(fused_norm).to(device))
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    topk = max(1, min(int(topk), len(id_to_name)))
    idx = np.argsort(-probs)[:topk]
    print(f"source={source}")
    print(f"predicted={id_to_name[int(idx[0])]} ")
    print("topk=")
    for i in idx:
        print(f"  {id_to_name[int(i)]}: {float(probs[int(i)]):.4f}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "predict"], default="train")
    p.add_argument("--features-dir", default="features")
    p.add_argument("--labels-csv", default="downloaded_authors.csv")
    p.add_argument("--label-column", default="canonical_composer")
    p.add_argument("--split-ratio", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--embed-ckpt", default="models/cnn_transformer_composer.pt")
    p.add_argument("--target-frames", type=int, default=1024)
    p.add_argument("--require-handcrafted", action="store_true")

    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--vae-epochs", type=int, default=50)
    p.add_argument("--clf-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    p.add_argument("--out", default="models/vae_fusion_classifier.pt")
    p.add_argument("--run-dir", default="runs")
    p.add_argument("--run-name", default="")

    p.add_argument("--predict-wav", default=None)
    p.add_argument("--predict-npz", default=None)
    p.add_argument("--topk", type=int, default=3)

    args = p.parse_args()

    run_dir = create_run_dir(
        base_dir=resolve_from_root(args.run_dir),
        script_name="vae_fusion_classifier",
        run_name=str(args.run_name) if args.run_name else None,
    )

    device = _get_device(args.device)

    if args.mode == "predict":
        return predict(
            model_path=resolve_from_root(args.out),
            ckpt_path=resolve_from_root(args.embed_ckpt),
            wav_path=resolve_from_root(args.predict_wav) if args.predict_wav else None,
            npz_path=resolve_from_root(args.predict_npz) if args.predict_npz else None,
            target_frames=int(args.target_frames),
            topk=int(args.topk),
            device=device,
        )

    return train_pipeline(
        features_dir=resolve_from_root(args.features_dir),
        labels_csv=resolve_from_root(args.labels_csv),
        label_column=str(args.label_column),
        ckpt_path=resolve_from_root(args.embed_ckpt),
        target_frames=int(args.target_frames),
        split_ratio=float(args.split_ratio),
        seed=int(args.seed),
        require_handcrafted=bool(args.require_handcrafted),
        latent_dim=int(args.latent_dim),
        beta=float(args.beta),
        vae_epochs=int(args.vae_epochs),
        clf_epochs=int(args.clf_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=device,
        out_path=resolve_from_root(args.out),
        run_dir=run_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
