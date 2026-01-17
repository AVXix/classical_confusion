# Download ~1GB subset of Maestro dataset

This repo includes a small helper to select and download approximately 1GB of audio files from the Maestro Kaggle dataset.

Prerequisites
- Install the Kaggle CLI and authenticate using ONE of these options:
	- **kaggle.json** (recommended): put your `kaggle.json` in `%USERPROFILE%\.kaggle\kaggle.json` on Windows.
	- **Environment variables**: set `KAGGLE_USERNAME` and `KAGGLE_KEY`.
	- **Custom file path**: pass `--kaggle-json "C:\\path\\to\\kaggle.json"`.

To get `kaggle.json`:
- Kaggle website → your profile → **Account** → **API** → **Create New Token** (downloads `kaggle.json`).

Place it in the default location (PowerShell):

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle" | Out-Null
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

Or set env vars (then open a new terminal):

```powershell
setx KAGGLE_USERNAME "<your_username>"
setx KAGGLE_KEY "<your_key>"
```
- Create a virtualenv and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Usage

Dry-run to see selected files (no downloads):

```powershell
python download_maestro_subset.py --outdir data --target-bytes 1073741824 --dry-run
```

If your `kaggle.json` is not in the default location:

```powershell
python download_maestro_subset.py --kaggle-json "C:\\path\\to\\kaggle.json" --outdir data --target-bytes 1073741824 --dry-run
```

To actually download the selected files:

```powershell
python download_maestro_subset.py --outdir data --target-bytes 1073741824
```

Notes
- The script selects files matching the extension (default `.wav`) in deterministic order until the cumulative size reaches the `--target-bytes` value.
- If a file lacks size metadata it will be skipped.
- You can change `--ext` to include other extensions (comma-separated).
