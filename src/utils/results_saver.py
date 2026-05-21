"""Persist training results to JSON so they can be reloaded across sessions."""
import json
import os
import glob as _glob


def save_result(result: dict, output_dir: str) -> str:
    """Serialise a result dict to <output_dir>/<run_id>/result.json.

    Returns the path written, or "" on failure.
    """
    run_id = result.get("run_id", "unknown_run")
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "result.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_make_serializable(result), f, indent=2, ensure_ascii=False)
        return path
    except Exception:
        return ""


def load_results(output_dir: str) -> list:
    """Scan <output_dir>/**/result.json and return a list of result dicts."""
    results = []
    if not os.path.isdir(output_dir):
        return results
    pattern = os.path.join(output_dir, "**", "result.json")
    for path in sorted(_glob.glob(pattern, recursive=True)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------

def _make_serializable(obj):
    """Recursively convert non-JSON-serialisable types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if hasattr(obj, "tolist"):          # numpy arrays / tensors
        return obj.tolist()
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)
