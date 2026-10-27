from pathlib import Path
import subprocess
import sys
from zipfile import BadZipFile, ZipFile

dataset = "planet-understanding-the-amazon-from-space"
download_dir = Path("data")
dataset_dir = download_dir / "planet"
zip_path = download_dir / f"{dataset}.zip"

download_dir.mkdir(parents=True, exist_ok=True)

def download_dataset():
    def has_files(folder: Path) -> bool:
        return folder.is_dir() and any(path.is_file() for path in folder.rglob("*"))

    if has_files(dataset_dir):
        print(f"Dataset already present in {dataset_dir}; skipping download. Empty this directory to redownload")
    else:
        result = subprocess.run(
            [
                sys.executable, "-m", "kaggle",
                "competitions", "download",
                "-c", dataset,
                "-p", str(download_dir),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            message = (result.stderr + result.stdout).lower()
            if any(term in message for term in ("unauthorized", "authentication", "401", "403")):
                raise RuntimeError(
                    "Kaggle authentication or competition access failed. "
                    "Check your API token and make sure you accepted the competition rules."
                )
            raise RuntimeError(f"Kaggle download failed:\n{result.stderr or result.stdout}")

        try:
            with ZipFile(zip_path) as archive:
                archive.extractall(dataset_dir)
        except (BadZipFile, FileNotFoundError) as exc:
            raise RuntimeError(f"Download did not produce a valid ZIP at {zip_path}") from exc

        print(f"Dataset downloaded and extracted to {dataset_dir}.")

if __name__ == "__main__":
    download_dataset()


