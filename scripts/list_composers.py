#!/usr/bin/env python3
"""List unique composers from maestro-v3.0.0.csv.

Writes full list to composer_names.txt.
"""

from __future__ import annotations

import csv
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        print(f"Missing file: {csv_path}")
        return 2

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("CSV has no headers")
            return 2

        fields_lower = {h.lower(): h for h in reader.fieldnames}
        composer_col = None
        for cand in ("composer", "canonical_composer", "canonical_composer_name", "composer_name"):
            if cand in fields_lower:
                composer_col = fields_lower[cand]
                break

        if composer_col is None:
            print("Could not find composer column.")
            print("Headers:")
            for h in reader.fieldnames:
                print(f"- {h}")
            return 2

        composers = set()
        for row in reader:
            v = (row.get(composer_col) or "").strip()
            if v:
                composers.add(v)

    composers_sorted = sorted(composers, key=lambda s: s.casefold())

    out_txt = root / "composer_names.txt"
    out_txt.write_text("\n".join(composers_sorted), encoding="utf-8")

    print(f"composer_col={composer_col}")
    print(f"unique_composers={len(composers_sorted)}")
    print(f"wrote={out_txt}")
    print("first_30=")
    for name in composers_sorted[:30]:
        print(name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

