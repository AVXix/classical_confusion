#!/usr/bin/env python3
"""Generate mel spectrogram images for WAV files using librosa.

Outputs one PNG per input WAV.

Example:
    ./.venv/Scripts/python.exe make_mel_spectrograms.py --in-dir data --out-dir mels
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="data", help="Folder containing .wav files")
    p.add_argument("--out-dir", default="mels", help="Folder to write .png images")
    p.add_argument("--sr", type=int, default=22050, help="Target sample rate for librosa.load")
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--fmax", type=int, default=8000)
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

    # Use non-interactive backend to avoid GUI requirements.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for i, wav_path in enumerate(wav_files, start=1):
        y, sr = librosa.load(wav_path, sr=args.sr, mono=True)
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            fmax=args.fmax,
            power=2.0,
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        out_png = out_dir / (wav_path.stem + "_mel.png")

        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        img = librosa.display.specshow(
            S_db,
            x_axis="time",
            y_axis="mel",
            sr=sr,
            hop_length=args.hop_length,
            fmax=args.fmax,
            ax=ax,
        )
        ax.set(title=f"Mel Spectrogram (dB): {wav_path.name}")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

        print(f"[{i}/{len(wav_files)}] wrote {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
