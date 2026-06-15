from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = RAW_DIR / "images"
METADATA_DIR = DATA_DIR / "metadata"

NACTI_METADATA_JSON_ZIP_PATH = METADATA_DIR / "nacti_metadata.1.14.json.zip"
NACTI_METADATA_JSON_PATH = METADATA_DIR / "nacti_metadata.1.14.json"
ALL_IMAGES_CSV_PATH = METADATA_DIR / "all_images.csv"
SUBSET_CSV_PATH = METADATA_DIR / "subset.csv"

NACTI_METADATA_JSON_ZIP_URL = (
    "https://storage.googleapis.com/public-datasets-lila/nacti/nacti_metadata.1.14.json.zip"
)

NACTI_IMAGE_BASE_URL = (
    "https://storage.googleapis.com/public-datasets-lila/nacti-unzipped/"
)

READABLE_LABELS = {
    "empty": "empty",
    "cervus elaphus": "red_deer",
    "sus scrofa": "wild_boar",
    "odocoileus hemionus": "mule_deer",
    "procyon lotor": "raccoon",
    "ursus americanus": "black_bear",
    "lynx rufus": "bobcat",
    "canis latrans": "coyote",
}

def ensure_directories() -> None:
    """
    Create the folders needed by the script.
    """

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url: str, output_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """
    Download a file from a URL to a local path.
    If the file already exists, it is not downloaded again.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Already exists: {output_path}")
        return

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as file:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=output_path.name,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))

def download_metadata() -> None:
    """
    Download and unzip the NACTI metadata JSON file.
    """
    print("Downloading NACTI metadata ZIP...")
    download_file(NACTI_METADATA_JSON_ZIP_URL, NACTI_METADATA_JSON_ZIP_PATH)

    if NACTI_METADATA_JSON_PATH.exists():
        print(f"Already extracted: {NACTI_METADATA_JSON_PATH}")
        return

    print("Extracting metadata JSON...")

    with zipfile.ZipFile(NACTI_METADATA_JSON_ZIP_PATH, "r") as zip_file:
        zip_file.extractall(METADATA_DIR)

    print(f"Extracted metadata to: {METADATA_DIR}")
def inspect_metadata() -> None:
    """
    Print basic information about the metadata JSON structure.
    """
    with open(NACTI_METADATA_JSON_PATH, "r") as file:
        metadata = json.load(file)

    print("Top-level keys:")
    print(metadata.keys())

    for key, value in metadata.items():
        if isinstance(value, list):
            print(f"{key}: list with {len(value):,} elements")
        else:
            print(f"{key}: {type(value)}")

    if "images" in metadata and len(metadata["images"]) > 0:
        print("\nExample image record:")
        print(metadata["images"][0])

    if "annotations" in metadata and len(metadata["annotations"]) > 0:
        print("\nExample annotation record:")
        print(metadata["annotations"][0])

    if "categories" in metadata and len(metadata["categories"]) > 0:
        print("\nExample category record:")
        print(metadata["categories"][0])

def metadata_json_to_dataframe() -> pd.DataFrame:
    """
    Convert COCO Camera Traps JSON metadata to a dataframe.

    Output columns:
    - image_id
    - file_name
    - label
    - category_id
    - image_url
    - local_path
    """
    with open(NACTI_METADATA_JSON_PATH, "r") as file:
        metadata = json.load(file)

    images = metadata["images"]
    annotations = metadata["annotations"]
    categories = metadata["categories"]

    image_id_to_file_name = {
        image["id"]: image["file_name"]
        for image in images
    }

    category_id_to_name = {
        category["id"]: category["name"]
        for category in categories
    }

    rows = []

    for annotation in annotations:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        file_name = image_id_to_file_name[image_id]
        label = category_id_to_name[category_id]
        image_url = NACTI_IMAGE_BASE_URL + file_name
        local_path = IMAGES_DIR / file_name
        rows.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "label": label,
                "category_id": category_id,
                "image_url": image_url,
                "local_path": str(local_path),
            }
        )
    df = pd.DataFrame(rows)
    return df

def create_all_images_csv() -> None:
    """
    Create data/metadata/all_images.csv from the NACTI JSON metadata.
    """
    if ALL_IMAGES_CSV_PATH.exists():
        print(f"Already exists: {ALL_IMAGES_CSV_PATH}")
        return

    print("Converting metadata JSON to CSV...")
    df = metadata_json_to_dataframe()

    df.to_csv(ALL_IMAGES_CSV_PATH, index=False)

    print(f"Saved: {ALL_IMAGES_CSV_PATH}")
    print(f"Rows: {len(df):,}")
    print("\nTop labels:")
    print(df["label"].value_counts().head(20))

def create_balanced_subset(
    classes: list[str],
    images_per_class: int,
    random_state: int = 42,
    output_csv_path: Path = SUBSET_CSV_PATH,
) -> pd.DataFrame:
    """
    Create a balanced subset with at most images_per_class examples per class.
    """
    df = pd.read_csv(ALL_IMAGES_CSV_PATH)

    subset_parts = []

    for class_name in classes:
        class_df = df[df["label"] == class_name]
        if class_df.empty:
            print(f"Warning: class not found: {class_name}")
            continue

        n = min(images_per_class, len(class_df))
        sampled_df = class_df.sample(
            n=n,
            random_state=random_state,
        )

        subset_parts.append(sampled_df)
        print(f"{class_name}: selected {n:,} / available {len(class_df):,}")

    if not subset_parts:
        raise ValueError("No classes were selected. Check class names.")

    subset_df = pd.concat(subset_parts, ignore_index=True)

    subset_df = subset_df.sample(
        frac=1,
        random_state=random_state,
    ).reset_index(drop=True)

    subset_df["readable_label"] = subset_df["label"].map(READABLE_LABELS).fillna(subset_df["label"])
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_csv(output_csv_path, index=False)

    print(f"\nSaved subset: {output_csv_path}")
    print(f"Total images selected: {len(subset_df):,}")

    return subset_df

def download_image(row: pd.Series, timeout: int = 30) -> bool:
    """
    Download one image from a dataframe row.

    Returns True if the image exists or was downloaded successfully.
    Returns False if the download failed.
    """
    url = row["image_url"]
    local_path = Path(row["local_path"])

    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        return True

    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

        return True

    except requests.RequestException as error:
        print(f"Failed: {url}")
        print(error)
        return False
    
def download_subset_images(input_csv_path: Path = SUBSET_CSV_PATH) -> None:
    """
    Download all images listed in the selected subset CSV.
    """
    subset_df = pd.read_csv(input_csv_path)

    success_count = 0
    failure_count = 0

    for _, row in tqdm(
        subset_df.iterrows(),
        total=len(subset_df),
        desc="Downloading images",
    ):
        success = download_image(row)

        if success:
            success_count += 1
        else:
            failure_count += 1

    print("\nDownload complete.")
    print(f"Successful: {success_count:,}")
    print(f"Failed: {failure_count:,}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a manageable NACTI camera-trap subset."
    )

    parser.add_argument(
        "--download-metadata",
        action="store_true",
        help="Download the NACTI metadata JSON.",
    )

    parser.add_argument(
        "--inspect-metadata",
        action="store_true",
        help="Inspect the structure of the metadata JSON.",
    )

    parser.add_argument(
        "--create-all-csv",
        action="store_true",
        help="Convert metadata JSON to data/metadata/all_images.csv.",
    )

    parser.add_argument(
        "--create-subset",
        action="store_true",
        help="Create a balanced subset CSV.",
    )

    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Download images from the selected subset CSV.",
    )

    parser.add_argument(
        "--subset-output",
        type=Path,
        default=SUBSET_CSV_PATH,
        help="CSV path where the balanced subset is saved.",
    )

    parser.add_argument(
        "--download-csv",
        type=Path,
        default=SUBSET_CSV_PATH,
        help="CSV path used when downloading images.",
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Class names to include in the subset.",
    )

    parser.add_argument(
        "--images-per-class",
        type=int,
        default=100,
        help="Maximum number of images per class.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    ensure_directories()

    if args.download_metadata:
        download_metadata()

    if args.inspect_metadata:
        inspect_metadata()

    if args.create_all_csv:
        create_all_images_csv()

    if args.create_subset:
        if args.classes is None:
            raise ValueError(
                "You must provide classes when using --create-subset."
            )

        create_balanced_subset(
            classes=args.classes,
            images_per_class=args.images_per_class,
            random_state=args.random_state,
            output_csv_path=args.subset_output,
        )

    if args.download_images:
        download_subset_images(input_csv_path=args.download_csv)


if __name__ == "__main__":
    main()