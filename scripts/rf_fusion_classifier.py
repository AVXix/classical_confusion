#!/usr/bin/env python3
"""RandomForest composer classifier using fused embeddings.

Final embedding = [CNN pooled features, Transformer pooled features, handcrafted features]

- CNN/Transformer act as feature extractors on X_spec (log-mel spectrogram)
- RandomForest does final classification

Inputs:
- features/*.npz created by scripts/extract_features.py
  - requires X_spec
  - optional X_hand (handcrafted stats) if you ran with --handcrafted
- downloaded_authors.csv created by scripts/authors_for_downloaded.py

Outputs:
- models/rf_fusion.joblib

Example:
    ./.venv/Scripts/python.exe scripts/rf_fusion_classifier.py --mode train --split-ratio 0.7
    ./.venv/Scripts/python.exe scripts/rf_fusion_classifier.py --mode predict --predict-wav data/some.wav
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow `from scripts import ...` when running this file directly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_from_root(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


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


@dataclass
class FeatureExtractorConfig:
    ckpt_path: Path
    target_frames: int


class TorchFeatureExtractor:
    """Loads the CNN/Transformer from a checkpoint and exposes pooled embeddings.

    Returns:
      - cnn_pooled: [128]
      - trans_pooled: [128]  (mean over time of transformer output)

    Note: this matches the current cnn_transformer in scripts/music_classfication.py,
    which aggregates transformer output by mean pooling.
    """

    def __init__(self, cfg: FeatureExtractorConfig):
        import torch
        from scripts import music_classfication as mc

        self._torch = torch
        self._mc = mc
        self.ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
        self.target_frames = int(cfg.target_frames)

        # Build the same model as the checkpoint for feature extraction.
        # We'll re-use mc.build_model for weights loading.
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
        )
        self.model.load_state_dict(self.ckpt["state_dict"], strict=True)
        self.model.eval()

        # Also create a copy of the CNN front-end so we can access pooled CNN embeddings.
        # This is a bit hacky but stable for our code: the model has `front` in both cases.
        self.front = getattr(self.model, "front", None)
        if self.front is None:
            raise ValueError("Expected model to have attribute 'front' for CNN feature extraction")

        # Transformer encoder is only present for cnn_transformer.
        self.is_transformer = str(self.ckpt.get("model", "cnn")) == "cnn_transformer"
        self.encoder = getattr(self.model, "encoder", None)
        self.pos = getattr(self.model, "pos", None)

    def embed(self, x_spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        x_spec = _pad_or_crop_time(x_spec, int(self.ckpt.get("target_frames", self.target_frames)))
        xb = torch.from_numpy(x_spec[None, None, :, :])  # [1, 1, F, T]

        with torch.no_grad():
            z = self.front(xb)  # [1, C=128, F', T]
            z_seq = z.mean(dim=2)  # [1, 128, T]
            cnn_pooled = z_seq.mean(dim=2).squeeze(0).cpu().numpy()  # [128]

            if self.is_transformer and self.encoder is not None and self.pos is not None:
                tbe = z_seq.permute(2, 0, 1)  # [T, 1, 128]
                tbe = self.pos(tbe)
                tbe = self.encoder(tbe)  # [T, 1, 128]
                trans_pooled = tbe.mean(dim=0).squeeze(0).cpu().numpy()  # [128]
            else:
                # If using cnn-only model, treat transformer embedding as same as cnn pooled.
                trans_pooled = cnn_pooled.copy()

        return cnn_pooled.astype(np.float32), trans_pooled.astype(np.float32)


def _split_indices(n: int, *, split_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("Need at least 2 samples to split")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    split_n = int(split_ratio * n)
    split_n = max(1, min(split_n, n - 1))
    return idx[:split_n], idx[split_n:]


def build_fused_vector(
    *,
    cnn_pooled: np.ndarray,
    trans_pooled: np.ndarray,
    x_hand: np.ndarray | None,
    require_handcrafted: bool,
) -> np.ndarray:
    if require_handcrafted and x_hand is None:
        raise ValueError(
            "X_hand is missing. Re-run feature extraction with: scripts/extract_features.py --handcrafted"
        )
    if x_hand is None:
        x_hand = np.zeros((47,), dtype=np.float32)

    fused = np.concatenate([cnn_pooled, trans_pooled, x_hand.astype(np.float32)], axis=0)
    return fused.astype(np.float32)


def train_rf(
    *,
    features_dir: Path,
    labels_csv: Path,
    ckpt_path: Path,
    split_ratio: float,
    seed: int,
    target_frames: int,
    require_handcrafted: bool,
    n_estimators: int,
    max_depth: int | None,
    out_path: Path,
) -> int:
    from sklearn.ensemble import RandomForestClassifier

    labels = _read_labels(labels_csv)
    items = _build_dataset_index(features_dir, labels)
    to_id, id_to_name = _make_label_vocab(items)

    extractor = TorchFeatureExtractor(FeatureExtractorConfig(ckpt_path=ckpt_path, target_frames=target_frames))

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for npz_path, composer in items:
        x_spec, x_hand = _load_npz(npz_path)
        cnn_p, tx_p = extractor.embed(x_spec)
        fused = build_fused_vector(
            cnn_pooled=cnn_p,
            trans_pooled=tx_p,
            x_hand=x_hand,
            require_handcrafted=require_handcrafted,
        )
        X_list.append(fused)
        y_list.append(int(to_id[composer]))

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)

    train_idx, test_idx = _split_indices(len(items), split_ratio=split_ratio, seed=seed)

    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=None if max_depth in (None, 0) else int(max_depth),
        random_state=int(seed),
    )
    clf.fit(X[train_idx], y[train_idx])

    acc = float(clf.score(X[test_idx], y[test_idx]))
    print(f"train_samples={len(train_idx)}")
    print(f"test_samples={len(test_idx)}")
    print(f"test_acc={acc:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "rf": clf,
            "id_to_name": id_to_name,
            "feature_dim": int(X.shape[1]),
            "split_ratio": float(split_ratio),
            "seed": int(seed),
            "ckpt_path": str(ckpt_path),
            "require_handcrafted": bool(require_handcrafted),
        },
        out_path,
    )
    print(f"saved={out_path}")
    return 0


def predict_rf(
    *,
    model_path: Path,
    ckpt_path: Path,
    predict_wav: Path | None,
    predict_npz: Path | None,
    target_frames: int,
    topk: int,
) -> int:
    bundle = joblib.load(model_path)
    clf = bundle["rf"]
    id_to_name: list[str] = bundle["id_to_name"]
    require_handcrafted = bool(bundle.get("require_handcrafted", False))

    extractor = TorchFeatureExtractor(FeatureExtractorConfig(ckpt_path=ckpt_path, target_frames=target_frames))

    if (predict_wav is None) == (predict_npz is None):
        raise ValueError("Provide exactly one of --predict-wav or --predict-npz")

    if predict_npz is not None:
        x_spec, x_hand = _load_npz(predict_npz)
        source = predict_npz.name
    else:
        x_spec = _extract_xspec_from_wav(predict_wav)
        x_hand = None
        source = predict_wav.name

    cnn_p, tx_p = extractor.embed(x_spec)
    fused = build_fused_vector(
        cnn_pooled=cnn_p,
        trans_pooled=tx_p,
        x_hand=x_hand,
        require_handcrafted=require_handcrafted,
    )

    probs = clf.predict_proba(fused[None, :])[0]
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
    p.add_argument("--features-dir", default="features")
    p.add_argument("--labels-csv", default="downloaded_authors.csv")
    p.add_argument("--split-ratio", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--embed-ckpt",
        default="models/cnn_transformer_composer.pt",
        help="Checkpoint for CNN/Transformer feature extractor",
    )
    p.add_argument("--target-frames", type=int, default=1024)
    p.add_argument(
        "--require-handcrafted",
        action="store_true",
        help="Require X_hand in npz (run extract_features.py --handcrafted). If not set, missing X_hand becomes zeros.",
    )

    p.add_argument("--rf-n-estimators", type=int, default=200)
    p.add_argument("--rf-max-depth", type=int, default=0, help="0 = None")

    p.add_argument("--out", default="models/rf_fusion.joblib")

    p.add_argument("--predict-wav", default=None)
    p.add_argument("--predict-npz", default=None)
    p.add_argument("--topk", type=int, default=3)

    args = p.parse_args()

    features_dir = resolve_from_root(args.features_dir)
    labels_csv = resolve_from_root(args.labels_csv)
    ckpt_path = resolve_from_root(args.embed_ckpt)
    out_path = resolve_from_root(args.out)

    if args.mode == "train":
        return train_rf(
            features_dir=features_dir,
            labels_csv=labels_csv,
            ckpt_path=ckpt_path,
            split_ratio=float(args.split_ratio),
            seed=int(args.seed),
            target_frames=int(args.target_frames),
            require_handcrafted=bool(args.require_handcrafted),
            n_estimators=int(args.rf_n_estimators),
            max_depth=None if int(args.rf_max_depth) == 0 else int(args.rf_max_depth),
            out_path=out_path,
        )

    predict_wav = resolve_from_root(args.predict_wav) if args.predict_wav else None
    predict_npz = resolve_from_root(args.predict_npz) if args.predict_npz else None

    return predict_rf(
        model_path=out_path,
        ckpt_path=ckpt_path,
        predict_wav=predict_wav,
        predict_npz=predict_npz,
        target_frames=int(args.target_frames),
        topk=int(args.topk),
    )


if __name__ == "__main__":
    raise SystemExit(main())
