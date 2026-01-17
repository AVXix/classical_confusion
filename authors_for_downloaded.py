#!/usr/bin/env python3
"""Find composer/author names for the WAV files you downloaded.

Matches each downloaded WAV basename to maestro-v3.0.0.csv's audio_filename.

Outputs:
- downloaded_authors.csv (per-file mapping)
- downloaded_unique_authors.txt (unique canonical_composer values)
"""

from __future__ import annotations

import csv
from pathlib import Path


def main() -> int:
    root = Path(r"d:\music classification")
    data_dir = root / "data"
    meta_csv = root / "maestro-v3.0.0.csv"

    if not data_dir.exists():
        print(f"Missing data dir: {data_dir}")
        return 2
    if not meta_csv.exists():
        print(f"Missing metadata CSV: {meta_csv}")
        return 2

    downloaded = sorted([p for p in data_dir.glob("*.wav") if p.is_file()])
    if not downloaded:
        print(f"No .wav files found in: {data_dir}")
        return 2

    # Build lookup from audio basename -> metadata row
    with meta_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("Metadata CSV has no headers")
            return 2

        required = ["canonical_composer", "canonical_title", "year", "audio_filename"]
        for r in required:
            if r not in reader.fieldnames:
                print(f"Missing column {r} in metadata CSV")
                print("Headers:")
                for h in reader.fieldnames:
                    print(f"- {h}")
                return 2

        by_audio_basename: dict[str, dict] = {}
        for row in reader:
            audio = (row.get("audio_filename") or "").strip()
            if not audio:
                continue
            base = Path(audio).name
            # Keep first match (should be unique in MAESTRO)
            by_audio_basename.setdefault(base, row)

    out_rows = []
    missing = []
    for p in downloaded:
        base = p.name
        row = by_audio_basename.get(base)
        if not row:
            missing.append(base)
            continue
        out_rows.append(
            {
                "downloaded_file": base,
                "canonical_composer": row.get("canonical_composer", ""),
                "canonical_title": row.get("canonical_title", ""),
                "year": row.get("year", ""),
                "audio_filename": row.get("audio_filename", ""),
            }
        )

    out_csv = root / "downloaded_authors.csv"
    out_txt = root / "downloaded_unique_authors.txt"

    if out_rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "downloaded_file",
                    "canonical_composer",
                    "canonical_title",
                    "year",
                    "audio_filename",
                ],
            )
            w.writeheader()
            w.writerows(out_rows)

        unique = sorted({r["canonical_composer"].strip() for r in out_rows if r["canonical_composer"]}, key=lambda s: s.casefold())
        out_txt.write_text("\n".join(unique), encoding="utf-8")

        print(f"matched_files={len(out_rows)}/{len(downloaded)}")
        print(f"unique_authors={len(unique)}")
        print(f"wrote={out_csv}")
        print(f"wrote={out_txt}")
        print("authors=")
        for name in unique:
            print(name)

    if missing:
        print("\nUnmatched downloaded files (no row found in maestro-v3.0.0.csv):")
        for m in missing:
            print(m)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
