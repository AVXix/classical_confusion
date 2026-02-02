#!/usr/bin/env python3
"""Find composer/author names for the WAV files you downloaded.

Matches each downloaded WAV basename to maestro-v3.0.0.csv's audio_filename.

Outputs:
- downloaded_authors.csv (per-file mapping)
- downloaded_unique_authors.txt (unique canonical_composer values)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


COMPOSER_ERA: dict[str, str] = {
    "Johann Sebastian Bach": "Baroque",
    "George Frideric Handel": "Baroque",
    "Domenico Scarlatti": "Baroque",
    "Joseph Haydn": "Classical",
    "Wolfgang Amadeus Mozart": "Classical",
    "Ludwig van Beethoven": "Romantic",
    "Franz Schubert": "Romantic",
    "Frédéric Chopin": "Romantic",
    "Franz Liszt": "Romantic",
    "Robert Schumann": "Romantic",
    "Johannes Brahms": "Romantic",
    "Pyotr Ilyich Tchaikovsky": "Romantic",
    "Sergei Rachmaninoff": "Late Romantic",
    "Claude Debussy": "Impressionist",
    "Alexander Scriabin": "Modern",
    "Mily Balakirev": "Romantic",
    "Modest Mussorgsky": "Romantic",
    "Isaac Albéniz": "Romantic",
}


def resolve_from_root(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def normalize_composer(name: str) -> str:
    """Apply project-specific composer normalization rules."""
    name = (name or "").strip()
    replacements = {
        "Johann Christian Fischer / Wolfgang Amadeus Mozart": "Wolfgang Amadeus Mozart",
        "Johann Sebastian Bach / Egon Petri": "Johann Sebastian Bach",
        "Johann Sebastian Bach / Ferruccio Busoni": "Johann Sebastian Bach",
        "Pyotr Ilyich Tchaikovsky / Mikhail Pletnev": "Pyotr Ilyich Tchaikovsky",
        "Pyotr Ilyich Tchaikovsky / Sergei Rachmaninoff": "Pyotr Ilyich Tchaikovsky",
        "Franz Schubert / Franz Liszt": "Franz Schubert",
    }
    normalized = replacements.get(name, name)
    return normalized


def composer_era_for(composer: str) -> str:
    composer = (composer or "").strip()
    if not composer:
        return ""
    # Per request: leave blank if we don't have an era mapping for this composer.
    return COMPOSER_ERA.get(composer, "")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data", help="Folder containing downloaded .wav files")
    p.add_argument("--meta-csv", default="maestro-v3.0.0.csv", help="MAESTRO metadata CSV")
    p.add_argument("--out-csv", default="downloaded_authors.csv", help="Output labels CSV")
    p.add_argument(
        "--out-unique-authors",
        default="downloaded_unique_authors.txt",
        help="Output text file with unique composers",
    )
    p.add_argument("--recursive", action="store_true", help="Scan data dir recursively for .wav")
    args = p.parse_args()

    data_dir = resolve_from_root(args.data_dir)
    meta_csv = resolve_from_root(args.meta_csv)
    out_csv = resolve_from_root(args.out_csv)
    out_txt = resolve_from_root(args.out_unique_authors)

    if not data_dir.exists():
        print(f"Missing data dir: {data_dir}")
        return 2
    if not meta_csv.exists():
        print(f"Missing metadata CSV: {meta_csv}")
        return 2

    if args.recursive:
        downloaded = sorted([p for p in data_dir.rglob("*.wav") if p.is_file()])
    else:
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

        required = ["canonical_composer", "canonical_title", "year", "audio_filename", "midi_filename"]
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

    out_rows: list[dict[str, str]] = []
    missing: list[str] = []
    for p in downloaded:
        base = p.name
        row = by_audio_basename.get(base)
        if not row:
            missing.append(base)
            out_rows.append(
                {
                    "downloaded_file": base,
                    "canonical_composer": "",
                    "composer_era": "",
                    "canonical_title": "",
                    "year": "",
                    "split": "",
                    "duration": "",
                    "audio_filename": "",
                    "midi_filename": "",
                }
            )
            continue
        composer = normalize_composer(row.get("canonical_composer", ""))
        out_rows.append(
            {
                "downloaded_file": base,
                "canonical_composer": composer,
                "composer_era": composer_era_for(composer),
                "canonical_title": row.get("canonical_title", ""),
                "year": row.get("year", ""),
                "split": row.get("split", ""),
                "duration": row.get("duration", ""),
                "audio_filename": row.get("audio_filename", ""),
                "midi_filename": row.get("midi_filename", ""),
            }
        )

    if out_rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "downloaded_file",
                    "canonical_composer",
                    "composer_era",
                    "canonical_title",
                    "year",
                    "split",
                    "duration",
                    "audio_filename",
                    "midi_filename",
                ],
            )
            w.writeheader()
            w.writerows(out_rows)

        unique = sorted(
            {r["canonical_composer"].strip() for r in out_rows if r["canonical_composer"]},
            key=lambda s: s.casefold(),
        )
        out_txt.write_text("\n".join(unique), encoding="utf-8")

        print(f"matched_files={len(out_rows)}/{len(downloaded)}")
        print(f"unique_authors={len(unique)}")
        print(f"wrote={out_csv}")
        print(f"wrote={out_txt}")
        # Avoid printing a huge author list when dataset is large.
        if len(unique) <= 50:
            print("authors=")
            for name in unique:
                print(name)
        else:
            print("authors= (suppressed; too many)")

    if missing:
        print("\nUnmatched downloaded files (no row found in maestro-v3.0.0.csv):")
        for m in missing[:50]:
            print(m)
        if len(missing) > 50:
            print(f"... and {len(missing) - 50} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

