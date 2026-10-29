from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

import tensorflow
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input


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
                "Kaggle authentication failed. "
                "Check your API token"
            ) from exc

def find_paths():
    classes_dir = next(dataset_dir.rglob("train_classes.csv"))
    train_dir = next(p for p in dataset_dir.rglob("train-jpg") if p.is_dir())
    test_dir = next(p for p in dataset_dir.rglob("test-jpg") if p.is_dir())
    return classes_dir, train_dir, test_dir

def load_data():
    classes_dir, train_dir, _ = find_paths()
    df = pd.read_csv(classes_dir)
    df["tags"] = df["tags"].str.split()
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["tags"]).astype("float32")
    paths = (train_dir / (df["image_name"] + ".jpg")).astype(str)

    train_p, val_p, train_y, val_y = train_test_split(
        paths, y, test_size=0.2
    )

    return train_p, val_p, train_y, val_y

def main():
    download_dataset()
    train_p, val_p, train_y, val_y = load_data()

if __name__ == "__main__":
    main()


