"""Generate and read label CSV files mapping time-frame folders to PB concentrations.

One row per time frame (folder) — all images inside share the same label, so
listing each image individually is redundant.  The num_images column records
how many image files are present in the folder at export time.
"""
import os
import csv


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def export_labels(experiments: list, output_path: str, class_cfg: dict = None) -> int:
    """Write a label CSV (one row per time frame).  Returns the number of rows written."""
    use_cls = class_cfg and class_cfg.get("enabled", False)
    classes = class_cfg.get("classes", []) if use_cls else []
    fieldnames = ["experiment", "time_frame", "folder_path",
                  "pb_concentration", "num_images", "split"]
    if use_cls:
        fieldnames.append("class_label")
    rows = _build_rows(experiments, classes if use_cls else None)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _pb_to_class(pb: float, classes: list) -> int:
    for i, cls in enumerate(classes[:-1]):
        max_val = cls.get("max")
        if max_val is not None and pb < float(max_val):
            return i
    return len(classes) - 1


def _count_images(folder: str) -> int:
    try:
        return sum(
            1 for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
    except OSError:
        return 0


def _build_rows(experiments: list, classes: list = None) -> list:
    rows = []
    for exp in experiments:
        exp_id = exp.get("id", "")
        split = exp.get("split", "train")
        for tf in exp.get("time_frames", []):
            folder = tf.get("folder_path", "")
            if not folder or not os.path.isdir(folder):
                continue
            pb = tf.get("pb_concentration", 0.0)
            n_imgs = _count_images(folder)
            row = {
                "experiment":      exp_id,
                "time_frame":      tf.get("name", ""),
                "folder_path":     folder,
                "pb_concentration": pb,
                "num_images":      n_imgs,
                "split":           split,
            }
            if classes:
                row["class_label"] = _pb_to_class(float(pb), classes)
            rows.append(row)
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
