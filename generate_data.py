import glob
import os
from pathlib import Path

import cv2
import pandas as pd


def generate_data():

    video_path = Path("data/raw")

    if not video_path.exists():
        raise RuntimeError(f"Video Path not found: {video_path}")

    images_out_dir = Path("data/images")
    labels_out_dir = Path("data/labels")

    # remove existing images and labels
    print(f"cleaning up images folder..")
    existing_images = glob.glob(str(images_out_dir) + "/*")
    for f in existing_images:
        os.remove(f)

    existing_labels = glob.glob(str(labels_out_dir) + "/*")
    for l in existing_labels:
        os.remove(l)

    # create output folders if necessary
    images_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    video_paths = glob.glob(str(video_path) + "/*")
    for v in video_paths:
        if not v.endswith(".csv"):
            process_vid(Path(v), images_out_dir)

    label_paths = glob.glob(str(video_path) + "/*.csv")
    for l in label_paths:
        process_labels(Path(l), labels_out_dir)


def process_labels(csv_path: Path, labels_out_dir: Path):

    output_file = labels_out_dir / "labels.csv"
    # stem returns the filename
    filename = csv_path.stem
    print(f"Processing CSV: {csv_path.name}")

    df = pd.read_csv(csv_path)
    frame_col_name = "frame"
    df_frame = df[frame_col_name]
    classes_col_name = [c for c in df.columns if c != frame_col_name]
    df_classes = df[classes_col_name]

    df_classes.insert(0, "filename", [f"{filename}_{idx}.jpg" for idx in df_frame], allow_duplicates=False)
    # append the content if exists, otherwise create
    df_classes.to_csv(output_file, mode="a", header=not output_file.exists(), index=False)


def process_vid(vid_path: Path, images_out_dir: Path):
    # stem returns the filename
    vid_prefix = vid_path.stem
    print(f"Processing video: {vid_path.name}")

    cap = cv2.VideoCapture(vid_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {vid_path}")
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(
            os.path.join(images_out_dir, f"{vid_prefix}_{frame_idx}.jpg"), frame
        )
        frame_idx += 1


generate_data()
