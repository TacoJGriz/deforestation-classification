
from pathlib import Path
from collections import Counter
import textwrap

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Dataset paths
BASE_DIR = Path("data/planets-dataset/planet/planet")
CSV_PATH = BASE_DIR / "train_classes.csv"
IMAGE_DIR = BASE_DIR / "train-jpg"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load image labels
df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Check available training images
available_images = {
    image.stem for image in IMAGE_DIR.glob("*.jpg")
}

available_df = df[
    df["image_name"].isin(available_images)
].copy()

missing_count = len(df) - len(available_df)

print("\nDataset image check:")
print("Total labeled images:", len(df))
print("Available training images:", len(available_df))
print("Missing labeled images:", missing_count)

if available_df.empty:
    raise RuntimeError(
        "No labeled training images were found. "
        "Check the train-jpg folder."
    )

# Count the different labels
label_counts = Counter()

for tags in df["tags"]:
    label_counts.update(tags.split())

label_counts = dict(
    sorted(
        label_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )
)

print("\nNumber of unique labels:", len(label_counts))
print("\nLabel distribution:")

for label, count in label_counts.items():
    print(f"{label}: {count}")

# Plot label distribution
plt.figure(figsize=(13, 6))
plt.bar(label_counts.keys(), label_counts.values())
plt.xticks(rotation=55, ha="right")
plt.title("Amazon Satellite Image Label Distribution")
plt.xlabel("Satellite Image Labels")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "label_distribution.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# Display sample satellite images
samples = available_df.sample(
    n=min(9, len(available_df)),
    random_state=42
)

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for ax in axes.flat:
    ax.axis("off")

for ax, (_, row) in zip(axes.flat, samples.iterrows()):
    image_path = IMAGE_DIR / f"{row['image_name']}.jpg"

    with Image.open(image_path) as image:
        ax.imshow(image.convert("RGB"))

    labels = row["tags"].replace(" ", ", ")

    ax.text(
    0.5, -0.08,
    textwrap.fill(labels, width=22),
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=8
)

plt.suptitle(
    "Sample Amazon Satellite Images and Labels",
    fontsize=16,
    y=0.98
)

plt.subplots_adjust(
    left=0.05,
    right=0.95,
    top=0.94,
    bottom=0.05,
    hspace=0.45,
    wspace=0.25
)

plt.savefig(
    OUTPUT_DIR / "sample_images.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("\nEDA completed!")
print("Graphs saved inside the results folder.")
print("1. label_distribution.png")
print("2. sample_images.png")
