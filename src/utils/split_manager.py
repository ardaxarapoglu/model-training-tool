"""Utilities for saving, loading, and applying experiment split configurations.

A split file is a JSON with this schema:
  {
    "name":    "descriptive name",
    "created": "2026-05-29 14:30:00",
    "splits": {
      "D03": {
        "experiment_split": "train",
        "time_frames": {"Y1": "train", "Y2": "validation", ...}
      },
      ...
    }
  }

A result JSON produced by the trainer also contains the same structure under
the key ``split_snapshot``, so a split can be loaded directly from a result
file as well as from a dedicated split file.
"""
import copy
import datetime
import json
import os


# ---------------------------------------------------------------------------
# Extract / Apply
# ---------------------------------------------------------------------------

def extract_split(experiments: list) -> dict:
    """Return a compact split-assignment dict from an experiments config list."""
    result = {}
    for exp in experiments:
        exp_split = exp.get("split", "train")
        tfs = {}
        for tf in exp.get("time_frames", []):
            tfs[tf["name"]] = tf.get("split", exp_split)
        result[exp["id"]] = {
            "experiment_split": exp_split,
            "time_frames": tfs,
        }
    return result


def apply_split(experiments: list, splits: dict) -> list:
    """Return a deep-copy of *experiments* with splits from *splits* applied.

    *splits* is the ``splits`` sub-dict (i.e. already unwrapped from the
    split-file envelope or from ``split_snapshot``).
    """
    result = copy.deepcopy(experiments)
    for exp in result:
        eid = exp["id"]
        if eid not in splits:
            continue
        entry = splits[eid]
        exp["split"] = entry.get("experiment_split", exp.get("split", "train"))
        tf_map = entry.get("time_frames", {})
        for tf in exp.get("time_frames", []):
            if tf["name"] in tf_map:
                tf["split"] = tf_map[tf["name"]]
    return result


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_split(experiments: list, path: str, name: str = "") -> None:
    """Serialise the current split assignments to *path* (JSON)."""
    data = {
        "name":    name or os.path.splitext(os.path.basename(path))[0],
        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "splits":  extract_split(experiments),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_split_file(path: str) -> dict:
    """Load a split file *or* a result JSON and normalise to split-file format.

    Accepted inputs
    ---------------
    * A dedicated split file (has a ``"splits"`` key).
    * A result JSON produced by the trainer (has a ``"split_snapshot"`` key).

    Returns
    -------
    dict with keys ``name``, ``created``, ``splits``.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "splits" in data:
        return data  # native split-file format

    if "split_snapshot" in data:
        return {
            "name":    data.get("run_id", os.path.splitext(os.path.basename(path))[0]),
            "created": "",
            "splits":  data["split_snapshot"],
        }

    raise ValueError(
        f"'{path}' does not contain a valid split configuration.\n"
        "Expected a 'splits' key (split file) or 'split_snapshot' key (result JSON)."
    )
