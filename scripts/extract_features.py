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
- X_hand: [47]                      (handcrafted vector) if enabled
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import librosa


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_from_root(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def log_amplitude(x: np.ndarray, mode: str) -> np.ndarray:
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
    do_handcrafted: bool,
) -> dict:
    y, sr_actual = librosa.load(wav_path, sr=sr, mono=True)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    P = np.abs(D) ** 2

    mel_fb = librosa.filters.mel(
        sr=sr_actual,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel_power = mel_fb @ P

    X_spec = log_amplitude(mel_power, log_mode).astype(np.float32)
    out = {"X_spec": X_spec}

    if do_mfcc:
        mfcc = librosa.feature.mfcc(S=X_spec, n_mfcc=n_mfcc)
        mean = mfcc.mean(axis=1)
        var = mfcc.var(axis=1)
        X_mfcc = np.stack([mean, var], axis=1).astype(np.float32)
        out["X_mfcc"] = X_mfcc

    if do_handcrafted:
        out["X_hand"] = compute_handcrafted_features(y=y, sr=sr_actual).astype(np.float32)

    return out


def _safe_mean_std(x: np.ndarray) -> tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    m = float(np.mean(x))
    s = float(np.std(x))
    if not np.isfinite(m):
        m = 0.0
    if not np.isfinite(s):
        s = 0.0
    return m, s


def compute_handcrafted_features(*, y: np.ndarray, sr: int) -> np.ndarray:
    """Compute handcrafted audio features.

    Output vector (47 dims):
    - Tempo/beat stats: [tempo, mean_beat_interval, std_beat_interval] (3)
    - ZCR stats: [zcr_mean, zcr_std] (2)
    - Spectral descriptors: centroid/rolloff/flux mean+std (6)
    - Chroma: mean (12) + std (12)
    - Tonnetz: mean (6) + std (6)
    """
    # Tempo / beat statistics
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        beat_intervals = np.diff(beat_times)
        beat_mean, beat_std = _safe_mean_std(beat_intervals)
        tempo_scalar = float(np.asarray(tempo).reshape(-1)[0])
        tempo_f = tempo_scalar if np.isfinite(tempo_scalar) else 0.0
    except Exception:
        tempo_f, beat_mean, beat_std = 0.0, 0.0, 0.0

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean, zcr_std = _safe_mean_std(zcr)

    # Spectral descriptors
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    centroid_mean, centroid_std = _safe_mean_std(spectral_centroid)
    rolloff_mean, rolloff_std = _safe_mean_std(spectral_rolloff)

    # Spectral flux (simple proxy using centroid frame diffs, per your example)
    if spectral_centroid.shape[1] >= 2:
        flux = np.diff(spectral_centroid, axis=1) ** 2
    else:
        flux = np.zeros((spectral_centroid.shape[0], 0), dtype=np.float32)
    flux_mean, flux_std = _safe_mean_std(flux)

    spectral_vec = np.array(
        [
            centroid_mean,
            centroid_std,
            rolloff_mean,
            rolloff_std,
            flux_mean,
            flux_std,
        ],
        dtype=np.float32,
    )

    # Harmonic / tonal features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    y_harm = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    tonnetz_std = np.std(tonnetz, axis=1)

    hand = np.concatenate(
        [
            np.array([tempo_f, beat_mean, beat_std], dtype=np.float32),
            np.array([zcr_mean, zcr_std], dtype=np.float32),
            spectral_vec,
            chroma_mean.astype(np.float32),
            chroma_std.astype(np.float32),
            tonnetz_mean.astype(np.float32),
            tonnetz_std.astype(np.float32),
        ]
    )
    return hand


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
    p.add_argument(
        "--handcrafted",
        action="store_true",
        help="Compute handcrafted features: tempo/beat, ZCR, spectral, chroma, tonnetz (X_hand)",
    )
    p.add_argument("--max-files", type=int, default=0, help="0 = no limit")
    args = p.parse_args()

    in_dir = resolve_from_root(args.in_dir)
    out_dir = resolve_from_root(args.out_dir)
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
            do_handcrafted=bool(args.handcrafted),
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
                "has_handcrafted": int("X_hand" in feats),
                "hand_dim": int(feats["X_hand"].shape[0]) if "X_hand" in feats else 0,
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
