"""Generate and read label CSV files mapping image paths to PB concentrations."""
import os
import csv


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def export_labels(experiments: list, output_path: str) -> int:
    """Write a label CSV.  Returns the number of rows written."""
    rows = _build_rows(experiments)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "experiment",
                "time_frame",
                "time_interval",
                "pb_concentration",
                "pb_distribution",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _build_rows(experiments: list) -> list:
    rows = []
    for exp in experiments:
        exp_id = exp.get("id", "")
        split = exp.get("split", "train")
        for tf in exp.get("time_frames", []):
            folder = tf.get("folder_path", "")
            if not folder or not os.path.isdir(folder):
                continue
            pb = tf.get("pb_concentration", 0.0)
            pb_dist = tf.get("pb_distribution", 0.0)
            interval = tf.get("time_interval", "")
            for fname in sorted(os.listdir(folder)):
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                    rows.append(
                        {
                            "image_path": os.path.join(folder, fname),
                            "experiment": exp_id,
                            "time_frame": tf.get("name", ""),
                            "time_interval": interval,
                            "pb_concentration": pb,
                            "pb_distribution": pb_dist,
                            "split": split,
                        }
                    )
    return rows


def count_images(experiments: list) -> dict:
    """Return {split: count} dict of available images per split."""
    counts = {"train": 0, "test": 0, "validation": 0}
    for exp in experiments:
        split = exp.get("split", "train")
        for tf in exp.get("time_frames", []):
            folder = tf.get("folder_path", "")
            if folder and os.path.isdir(folder):
                n = sum(
                    1
                    for f in os.listdir(folder)
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTS
                )
                counts[split] = counts.get(split, 0) + n
    return counts
