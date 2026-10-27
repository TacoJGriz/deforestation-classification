from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

dataset = "nikitarom/planets-dataset"
dataset_dir = Path("data/planets-dataset")

def download_dataset():
    dataset_dir.mkdir(parents=True, exist_ok=True)

    def has_files(folder: Path) -> bool:
        return folder.is_dir() and any(path.is_file() for path in folder.rglob("*"))

    if has_files(dataset_dir):
        print(f"Dataset already present in {dataset_dir}; skipping download. Empty this directory to redownload")
    else:
        try:
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                dataset,
                path=str(dataset_dir),
                unzip=True,
                quiet=False
            )
        except Exception as exc:
            raise RuntimeError(
                "Kaggle authentication or competition access failed. "
                "Check your API token and make sure you accepted the competition rules."
            ) from exc


def main():
    download_dataset()

if __name__ == "__main__":
    main()


