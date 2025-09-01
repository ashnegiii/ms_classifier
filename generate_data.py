import glob
import os
from pathlib import Path

import cv2
import pandas as pd


def generate_data(video_dir: str, images_out_dir: str, labels_out_dir: str):
    """
    Extract frames and labels from a video folder.

    Args:
        video_dir (str): Path to the folder containing videos and CSV label files.
        images_out_dir (str): Path where extracted images will be stored.
        labels_out_dir (str): Path where processed labels will be stored.
    """

    video_path = Path(video_dir)

    if not video_path.exists():
        raise RuntimeError(f"Video Path not found: {video_path}. Please provide a valid path.")

    images_out_dir = Path(images_out_dir)
    labels_out_dir = Path(labels_out_dir)

    # cleanup old images and labels
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
        if not v.endswith(".csv"):
            process_vid(Path(v), images_out_dir)

    # process CSV labels
    label_paths = glob.glob(str(video_path) + "/*.csv")
    for l in label_paths:
        process_labels(Path(l), labels_out_dir)


def process_labels(csv_path: Path, labels_out_dir: Path):
    output_file = labels_out_dir / "labels.csv"
    filename = csv_path.stem
    print(f"Processing CSV: {csv_path.name}")

    df = pd.read_csv(csv_path)
    frame_col_name = "frame"
    df_frame = df[frame_col_name]
    classes_col_name = [c for c in df.columns if c.lower() != frame_col_name and c.lower() != 'timestamp']
    df_classes = df[classes_col_name]

    df_classes.insert(
        0, "filename", [f"{filename}_{idx}.jpg" for idx in df_frame], allow_duplicates=False
    )
    df_classes.to_csv(output_file, mode="a", header=not output_file.exists(), index=False)


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
    # Example usage
    generate_data(
        video_dir="data/raw",
        images_out_dir="data/images",
        labels_out_dir="data/labels"
    )
    
    #generate_data(
    #    video_dir="data/predict/03-04-17",
    #    images_out_dir="data/predict/03-04-17",
    #    labels_out_dir="data/predict/03-04-17",
    #)
