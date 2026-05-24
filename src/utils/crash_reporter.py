"""Lightweight crash reporter.

Hooks into sys.excepthook for uncaught GUI exceptions and exposes
log_worker_crash() for worker thread errors.  Both write a plain-text
report to ./crashes/ so failures are never silently lost.

Overhead when nothing goes wrong: zero — the hook is only called on crash.
"""
import sys
import os
import datetime
import traceback

_crash_dir: str = "./crashes"


def install(crash_dir: str = "./crashes") -> None:
    """Install as sys.excepthook.  Call once at app startup."""
    global _crash_dir
    _crash_dir = crash_dir
    sys.excepthook = _handle_uncaught


def log_worker_crash(run_id: str, tb_str: str) -> None:
    """Persist a worker-thread traceback (training / grid-search crash)."""
    _write(f"worker_{run_id}", f"Worker crash — run_id: {run_id}\n{'='*60}\n{tb_str}")


# ---------------------------------------------------------------------------

def _handle_uncaught(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    _write("crash", f"Uncaught exception\n{'='*60}\n{tb_str}")
    sys.__excepthook__(exc_type, exc_value, exc_tb)   # still print to stderr


def _write(prefix: str, body: str) -> None:
    try:
        os.makedirs(_crash_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(_crash_dir, f"{prefix}_{ts}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Timestamp : {datetime.datetime.now().isoformat()}\n")
            f.write(f"Python    : {sys.version}\n")
            f.write(body)
        print(f"[CrashReporter] Report written → {path}", file=sys.stderr)
    except Exception:
        pass   # never let the reporter itself crash anything
