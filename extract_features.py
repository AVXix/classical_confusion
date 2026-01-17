#!/usr/bin/env python3
"""Extract per-clip features from WAV files.

Main input (Mel-spectrogram):
1) STFT
2) Mel filterbank
3) Log amplitude

Optional MFCCs:
- compute MFCCs from log-mel
- store mean & variance per clip

Outputs per clip (saved as .npz):
- X_spec: [freq_bins x time_frames]  (freq_bins = n_mels)
- X_mfcc: [mfcc_dim x 2]            (columns: mean, var) if enabled

Example (PowerShell):
  & "D:/music classification/.venv/Scripts/python.exe" "d:/music classification/extract_features.py" \
    --in-dir "d:/music classification/data" --out-dir "d:/music classification/features" --mfcc
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import librosa


def log_amplitude(x: np.ndarray, mode: str) -> np.ndarray:
    # x is non-negative (power mel)
    if mode == "db":
        return librosa.power_to_db(x, ref=np.max)
    if mode == "log":
        return np.log(x + 1e-10)
    raise ValueError(f"Unknown log mode: {mode}")


def extract_one(
    wav_path: Path,
    *,
    sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int | None,
    n_mels: int,
    fmin: float,
    fmax: float | None,
    log_mode: str,
    do_mfcc: bool,
    n_mfcc: int,
) -> dict:
    y, sr_actual = librosa.load(wav_path, sr=sr, mono=True)

    # 1) STFT -> power spectrogram
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    P = np.abs(D) ** 2

    # 2) Mel filterbank
    mel_fb = librosa.filters.mel(
        sr=sr_actual,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel_power = mel_fb @ P

    # 3) Log amplitude
    X_spec = log_amplitude(mel_power, log_mode).astype(np.float32)

    out = {"X_spec": X_spec}

    if do_mfcc:
        # MFCCs from log-mel (in dB works best; log() also works)
        mfcc = librosa.feature.mfcc(S=X_spec, n_mfcc=n_mfcc)
        mean = mfcc.mean(axis=1)
        var = mfcc.var(axis=1)
        X_mfcc = np.stack([mean, var], axis=1).astype(np.float32)  # [mfcc_dim x 2]
        out["X_mfcc"] = X_mfcc

    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="data", help="Folder with .wav files")
    p.add_argument("--out-dir", default="features", help="Folder for .npz outputs")
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--win-length", type=int, default=0, help="0 = use n_fft")
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--fmin", type=float, default=0.0)
    p.add_argument("--fmax", type=float, default=8000.0)
    p.add_argument("--log", dest="log_mode", choices=["db", "log"], default="db")
    p.add_argument("--mfcc", action="store_true", help="Also compute MFCC mean/var per clip")
    p.add_argument("--n-mfcc", type=int, default=20)
    p.add_argument("--max-files", type=int, default=0, help="0 = no limit")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if not in_dir.exists():
        print(f"Input dir not found: {in_dir}")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted([p for p in in_dir.rglob("*.wav") if p.is_file()])
    if args.max_files and args.max_files > 0:
        wav_files = wav_files[: args.max_files]
    if not wav_files:
        print(f"No .wav files found under: {in_dir}")
        return 2

    win_length = args.n_fft if args.win_length in (0, None) else args.win_length
    fmax = None if args.fmax <= 0 else float(args.fmax)

    index_csv = out_dir / "features_index.csv"
    index_rows = []

    for i, wav_path in enumerate(wav_files, start=1):
        feats = extract_one(
            wav_path,
            sr=args.sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            win_length=win_length,
            n_mels=args.n_mels,
            fmin=float(args.fmin),
            fmax=fmax,
            log_mode=args.log_mode,
            do_mfcc=bool(args.mfcc),
            n_mfcc=int(args.n_mfcc),
        )

        out_path = out_dir / (wav_path.stem + ".npz")
        np.savez_compressed(out_path, **feats)

        X_spec = feats["X_spec"]
        index_rows.append(
            {
                "wav_file": wav_path.name,
                "out_npz": out_path.name,
                "spec_freq_bins": int(X_spec.shape[0]),
                "spec_time_frames": int(X_spec.shape[1]),
                "has_mfcc": int("X_mfcc" in feats),
            }
        )

        print(f"[{i}/{len(wav_files)}] wrote {out_path.name} | X_spec={X_spec.shape}")

    with index_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        w.writeheader()
        w.writerows(index_rows)

    print(f"Wrote index: {index_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
