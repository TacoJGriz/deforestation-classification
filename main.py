from pathlib import Path
import subprocess
import sys
from zipfile import BadZipFile, ZipFile

import time
import libtorrent as lt

dataset = "planet-understanding-the-amazon-from-space"
download_dir = Path("data")
dataset_dir = download_dir / "planet"
zip_path = download_dir / f"{dataset}.zip"

download_dir.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data/planet")

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

def download_torrent(torrent_name: str):
    torrent_path = DATA_DIR / torrent_name
    if not torrent_path.is_file():
        raise FileNotFoundError(f"Missing torrent file: {torrent_path}")

    info = lt.torrent_info(str(torrent_path))

    files = info.files()
    missing = [
        Path(DATA_DIR / files.file_path(i))
        for i in range(files.num_files())
        if not (DATA_DIR / files.file_path(i)).is_file()
        or (DATA_DIR / files.file_path(i)).stat().st_size != files.file_size(i)
    ]

    if not missing:
        print(f"{torrent_name}: files already present; skipping.")
        return

    session = lt.session()
    params = lt.add_torrent_params()
    params.ti = info
    params.save_path = str(DATA_DIR.resolve())
    handle = session.add_torrent(params)

    print(f"Downloading {torrent_name}...")
    while True:
        status = handle.status()
        print(
            f"\r{status.progress * 100:5.1f}% downloaded, "
            f"Rate: {status.download_rate / 1_000_000:.2f} MB/s, "
            f"Peers {status.num_peers}, "
            f"State: {status.state}",
            end="",
            flush=True,
        )
        if status.progress >= 1.0:
            break
        time.sleep(2)

    print(f"\nFinished {torrent_name}.")

def download_images():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_torrent("Kaggle-planet-train-tif.torrent")
    download_torrent("Kaggle-planet-test-tif.torrent")

def main():
    download_dataset()
    download_images()

if __name__ == "__main__":
    main()


