#!/usr/bin/env python3
"""
Download a subset (~target bytes) of files from a Kaggle dataset.

Usage:
  python scripts/download_maestro_subset.py --outdir data --target-bytes 1073741824

Requires: kaggle (place your `kaggle.json` as described in README)
"""
import os
import argparse
from pathlib import Path
import sys
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow importing from project root when running this file directly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_from_root(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def human(n):
    for unit in ['B','KB','MB','GB','TB']:
        if abs(n) < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob('*'):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _default_kaggle_json_path() -> Path:
    # Kaggle API defaults to ~/.kaggle/kaggle.json
    home = Path.home()
    return home / '.kaggle' / 'kaggle.json'


def _print_kaggle_setup_help(expected_path: Path) -> None:
    print('Could not find Kaggle credentials.')
    print(f'Expected kaggle.json at: {expected_path}')
    print('Fix options (Windows):')
    print(r'  1) Put kaggle.json here: %USERPROFILE%\.kaggle\kaggle.json')
    print(r'     (Create the folder if needed: mkdir %USERPROFILE%\.kaggle)')
    print('  2) Or pass a custom path:')
    print('       python scripts/download_maestro_subset.py --kaggle-json "C:\\path\\to\\kaggle.json" --dry-run')
    print('  3) Or set env vars (current session):')
    print('       setx KAGGLE_USERNAME "<your_username>"')
    print('       setx KAGGLE_KEY "<your_key>"')
    print('     Then open a new terminal and retry.')


def _get_file_size_bytes(f) -> int:
    # Kaggle SDK objects expose `total_bytes`.
    size = (
        getattr(f, 'total_bytes', None)
        or getattr(f, 'totalBytes', None)
        or getattr(f, 'size', None)
        or getattr(f, 'sizeInBytes', None)
    )
    try:
        return int(size) if size is not None else 0
    except (TypeError, ValueError):
        return 0


def _iter_dataset_files(api, dataset: str, page_size: int = 1000) -> Iterable:
    page_token = None
    while True:
        resp = api.dataset_list_files(dataset, page_token=page_token, page_size=page_size)
        for f in resp.files:
            yield f
        page_token = getattr(resp, 'nextPageToken', None) or getattr(resp, 'next_page_token', None)
        if not page_token:
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='alonhaviv/the-maestro-dataset-v3-0-0')
    p.add_argument('--outdir', default='data')
    p.add_argument('--target-bytes', type=int, default=1_000_000_000,
                   help='Approximate target total bytes to download')
    p.add_argument('--ext', default='.wav',
                   help='File extension to include (comma-separated)')
    p.add_argument('--kaggle-json', default=None,
                   help='Path to kaggle.json (overrides default ~/.kaggle/kaggle.json)')
    p.add_argument('--kaggle-config-dir', default=None,
                   help='Directory containing kaggle.json (sets KAGGLE_CONFIG_DIR)')
    p.add_argument('--dry-run', action='store_true', help='Only select files, do not download')
    p.add_argument('--show-selected', type=int, default=50,
                   help='When using --dry-run, print at most this many selected files (0 prints none)')
    p.add_argument('--list-files', action='store_true',
                   help='List files in the dataset and exit (useful to pick --ext)')
    p.add_argument('--enforce-target-bytes', action='store_true',
                   help='When Kaggle file sizes are missing, stop downloading once on-disk outdir size reaches target')
    p.add_argument('--page-size', type=int, default=1000,
                   help='Kaggle files API page size (higher reduces API calls; default 1000)')
    p.add_argument('--no-skip-existing', action='store_true',
                   help='Do not skip files already present in outdir')
    p.add_argument('--run-dir', default='runs',
                   help='Folder to write run_info.txt')
    p.add_argument('--run-name', default='',
                   help='Optional tag to include in run folder name')
    args = p.parse_args()

    from scripts.run_logging import create_run_dir, write_run_info_txt

    run_dir = create_run_dir(
        base_dir=resolve_from_root(args.run_dir),
        script_name='download_maestro_subset',
        run_name=str(args.run_name) if args.run_name else None,
    )

    outdir_path = resolve_from_root(args.outdir)
    os.makedirs(outdir_path, exist_ok=True)

    # Allow overriding where Kaggle looks for kaggle.json
    if args.kaggle_json:
        kaggle_json_path = Path(args.kaggle_json).expanduser().resolve()
        os.environ['KAGGLE_CONFIG_DIR'] = str(kaggle_json_path.parent)
    elif args.kaggle_config_dir:
        os.environ['KAGGLE_CONFIG_DIR'] = str(Path(args.kaggle_config_dir).expanduser().resolve())

    expected_kaggle_json = _default_kaggle_json_path()
    if 'KAGGLE_CONFIG_DIR' in os.environ:
        expected_kaggle_json = Path(os.environ['KAGGLE_CONFIG_DIR']) / 'kaggle.json'

    # If neither env vars nor kaggle.json exists, fail fast with clear instructions.
    has_env_creds = bool(os.environ.get('KAGGLE_USERNAME')) and bool(os.environ.get('KAGGLE_KEY'))
    if not has_env_creds and not expected_kaggle_json.exists():
        _print_kaggle_setup_help(expected_kaggle_json)
        raise SystemExit(2)

    # Import Kaggle lazily because the kaggle package may call sys.exit(1)
    # during import if credentials are missing.
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except SystemExit as e:
        if e.code == 1 and (not has_env_creds) and (not expected_kaggle_json.exists()):
            _print_kaggle_setup_help(expected_kaggle_json)
            raise SystemExit(2)
        raise
    except ModuleNotFoundError:
        print('Missing dependency: kaggle')
        print('Install it with: pip install -r requirements.txt')
        raise SystemExit(2)

    api = KaggleApi()
    api.authenticate()

    print(f"Listing files for dataset {args.dataset}...")
    files = list(_iter_dataset_files(api, args.dataset, page_size=args.page_size))

    write_run_info_txt(
        run_dir=run_dir,
        script_name='download_maestro_subset',
        argv=sys.argv,
        args=args,
        extra={
            'dataset_files_listed': len(files),
            'outdir': str(outdir_path),
            'current_on_disk_bytes': _dir_size_bytes(outdir_path),
        },
    )

    if args.list_files:
        for f in files:
            name = getattr(f, 'name', None)
            size = _get_file_size_bytes(f)
            print(f"{name} — {human(size)}")
        return

    exts = [e.strip().lower() for e in args.ext.split(',') if e.strip()]

    candidates = []
    for f in files:
        name = f.name
        lname = name.lower()
        if not any(lname.endswith(e) for e in exts):
            continue
        size = _get_file_size_bytes(f)
        candidates.append((name, size))

    if not candidates:
        print('No candidate files found with given extensions.')
        print('Tip: run with --ext .zip to include archives, or choose a different extension based on dataset contents.')
        return

    candidates.sort()

    skip_existing = not args.no_skip_existing
    existing = set()
    if skip_existing and outdir_path.exists():
        for pth in outdir_path.glob('*'):
            if pth.is_file():
                existing.add(pth.name)

    current_on_disk = _dir_size_bytes(outdir_path)
    if current_on_disk >= args.target_bytes:
        print(
            f"Outdir already meets target: {human(current_on_disk)} (target {human(args.target_bytes)}). Nothing to do."
        )
        return

    remaining_bytes = max(0, args.target_bytes - current_on_disk)

    selected = []
    total = 0
    for name, size in candidates:
        local_name = Path(name).name
        if skip_existing and local_name in existing:
            continue
        if total >= remaining_bytes:
            break
        selected.append((name, size))
        total += size

    if total == 0 and candidates and (not args.enforce_target_bytes):
        # If Kaggle reports missing sizes (0), still select a large batch to make progress.
        # (With enforce-target-bytes, we rely on on-disk size checks per file.)
        selected = []
        for name, size in candidates:
            local_name = Path(name).name
            if skip_existing and local_name in existing:
                continue
            selected.append((name, size))
            if len(selected) >= 5000:
                break

    print(
        f"Current on disk: {human(current_on_disk)} | Target total: {human(args.target_bytes)} | Remaining: {human(remaining_bytes)}"
    )
    print(f"Selected {len(selected)} files, planned download {human(total)}")

    if args.dry_run:
        limit = max(0, int(args.show_selected))
        to_show = [] if limit == 0 else selected[:limit]
        for name, size in to_show:
            print(f"{name} — {human(size)}")
        if limit != 0 and len(selected) > limit:
            print(f"... and {len(selected) - limit} more")
        return

    for name, size in selected:
        if args.enforce_target_bytes:
            on_disk = _dir_size_bytes(outdir_path)
            if on_disk >= args.target_bytes:
                print(f"Reached target on disk: {human(on_disk)} (target {human(args.target_bytes)}). Stopping.")
                break

        print(f"Downloading {name} ({human(size)})...")
        try:
            api.dataset_download_file(args.dataset, file_name=name, path=str(outdir_path), force=False)
        except Exception as e:
            print(f"Failed to download {name}: {e}")

    print('Done. Check the output directory for downloaded files.')


if __name__ == '__main__':
    main()
