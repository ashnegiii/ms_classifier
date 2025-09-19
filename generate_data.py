import glob
import os
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd


def generate_data(
    video_dir: str,
    images_out_dir: str,
    labels_out_dir: str,
    delete_old: bool = False,
    drop_empty_ratio: float = 0.0,   
    random_state: int = 42   
):
    """
    Extract frames and labels from a video folder.

    Args:
        video_dir (str): Path to the folder containing videos and CSV label files.
        images_out_dir (str): Path where extracted images will be stored.
        labels_out_dir (str): Path where processed labels will be stored.
        delete_old (bool): If True, removes old images/labels first.
        drop_empty_ratio (float): In [0.0, 1.0]. Proportion of rows to DROP where all classes are 0.
                                  0.0 keeps all empty rows; 1.0 drops all empty rows.
        random_state (int|None): Random seed for reproducible sampling.
    """
    if not (0.0 <= drop_empty_ratio <= 1.0):
        raise ValueError("drop_empty_ratio must be between 0.0 and 1.0")

    video_path = Path(video_dir)

    if not video_path.exists():
        raise RuntimeError(f"Video Path not found: {video_path}. Please provide a valid path.")

    images_out_dir = Path(images_out_dir)
    labels_out_dir = Path(labels_out_dir)

    # cleanup old images and labels
    if delete_old:
        print(f"Cleaning up {images_out_dir} and {labels_out_dir}...")
        for f in glob.glob(str(images_out_dir) + "/*.jpg"):
            os.remove(f)
    for l in glob.glob(str(labels_out_dir) + "/*.csv"):
        os.remove(l)

    # create output folders if necessary
    images_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    # process videos
    video_paths = glob.glob(str(video_path) + "/*")
    for v in video_paths:
        # check if there is no image with the prefix already
        if not glob.glob(str(images_out_dir / f"{Path(v).stem}_*.jpg")):
            if not v.endswith(".csv"):
                process_vid(Path(v), images_out_dir)

    # process CSV labels
    label_paths = glob.glob(str(video_path) + "/*.csv")
    for l in label_paths:
        # only write labels if not already present for this file prefix
        if not glob.glob(str(labels_out_dir / f"{Path(l).stem}_*.csv")):
            process_labels(
                csv_path=Path(l),
                labels_out_dir=labels_out_dir,
                drop_empty_ratio=drop_empty_ratio,
                random_state=random_state
            )


def process_labels(
    csv_path: Path,
    labels_out_dir: Path,
    drop_empty_ratio: float = 0.0,
    random_state: Optional[int] = None
):
    """
    Append labels for a single CSV into labels.csv, optionally down-sampling rows
    where all class columns == 0 by drop_empty_ratio.
    """
    output_file = labels_out_dir / "labels.csv"
    filename_prefix = csv_path.stem
    print(f"Processing CSV: {csv_path.name}")

    df = pd.read_csv(csv_path)

    frame_col_name = "frame"
    # class columns = everything except 'frame' and 'timestamp' (case-insensitive)
    classes_cols = [c for c in df.columns if c.lower() not in {frame_col_name, "timestamp"}]

    # Build the output table: filename + class columns
    df_out = df[classes_cols].copy()
    df_out.insert(
        0, "filename", [f"{filename_prefix}_{idx}.jpg" for idx in df[frame_col_name]], allow_duplicates=False
    )

    if drop_empty_ratio > 0.0:
        # Identify rows where all class labels are 0 (no character present)
        empty_mask = (df[classes_cols] == 0).all(axis=1)

        # Split into positives (at least one char) and empties
        pos_rows = df_out.loc[~empty_mask]
        empty_rows = df_out.loc[empty_mask]

        # Keep only a fraction of the empty rows (1 - drop_empty_ratio)
        keep_frac = 1.0 - drop_empty_ratio
        if keep_frac <= 0.0:
            kept_empty = empty_rows.iloc[0:0]  # keep none
        else:
            kept_empty = empty_rows.sample(frac=keep_frac, random_state=random_state) if len(empty_rows) > 0 else empty_rows

        # Recombine, preserving original order by index
        df_out = pd.concat([pos_rows, kept_empty]).sort_index()

    # Append to labels.csv (write header only if file doesn't exist yet)
    df_out.to_csv(output_file, mode="a", header=not output_file.exists(), index=False)


def process_vid(vid_path: Path, images_out_dir: Path):
    vid_prefix = vid_path.stem
    print(f"Processing video: {vid_path.name}")

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {vid_path}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(images_out_dir / f"{vid_prefix}_{frame_idx}.jpg"), frame)
        frame_idx += 1


if __name__ == "__main__":
    generate_data(
        video_dir="data/raw",
        images_out_dir="data/images",
        labels_out_dir="data/labels",
        delete_old=True,
        drop_empty_ratio=0.5,
        random_state=42
    )

    # For predict set (no dropping)
    # generate_data(
    #     video_dir="data/predict/03-04-17",
    #     images_out_dir="data/predict/03-04-17",
    #     labels_out_dir="data/predict/03-04-17",
    #     drop_empty_ratio=0.0
    # )
