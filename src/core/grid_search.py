"""Grid search orchestration.  Produces all param combinations and runs them sequentially."""
import itertools
import datetime
from typing import List, Dict

from qtpy.QtCore import QThread, Signal, Qt

from .trainer import TrainingWorker


def parse_multi_values(entry) -> list:
    """Return a list of candidate values for a parameter entry dict or scalar."""
    if isinstance(entry, dict):
        if entry.get("use_grid", False):
            raw = entry.get("values", "")
            parts = [p.strip() for p in str(raw).split(",") if p.strip()]
            return parts if parts else [entry.get("value", "")]
        return [entry.get("value", "")]
    return [str(entry)]


def generate_combinations(t_cfg: dict, model_cfg: dict) -> List[Dict]:
    """Return a list of resolved param dicts (one per grid-search combo).

    Architecture is now a standard _ParamRow in the training config so it
    participates in grid search exactly like batch_size, lr, etc.
    """
    param_keys = [
        "batch_size", "learning_rate", "optimizer",
        "weight_decay", "loss", "architecture",
    ]
    value_lists = [parse_multi_values(t_cfg.get(k, {})) for k in param_keys]
    combos = [dict(zip(param_keys, combo)) for combo in itertools.product(*value_lists)]
    return combos


class GridSearchWorker(QThread):
    run_started = Signal(int, int, dict)   # run_num, total, params
    run_finished = Signal(int, dict)       # run_num, result
    run_log = Signal(str)
    run_progress = Signal(int, int, int, int)  # run_num, total_runs, epoch, total_epochs
    epoch_metrics = Signal(dict)           # relayed from current TrainingWorker
    all_done = Signal(list)                # list of all results
    error = Signal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._stop = False
        self._current_worker = None

    def stop(self):
        self._stop = True
        if self._current_worker:
            self._current_worker.stop()

    def run(self):
        try:
            t_cfg = self.config["training"]
            model_cfg = self.config.get("model", {})
            combos = generate_combinations(t_cfg, model_cfg)
            total = len(combos)
            self.run_log.emit(f"Grid search: {total} combination(s) to evaluate.")

            # Timestamp prefix makes every grid-search session unique (no overwrites)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            results = []
            for i, params in enumerate(combos):
                if self._stop:
                    break
                run_id = f"gs_{ts}_{i + 1:03d}"
                self.run_started.emit(i + 1, total, params)
                self.run_log.emit(f"\n=== Run {i+1}/{total}: {params} ===")

                worker = TrainingWorker(self.config, run_params=params, run_id=run_id)
                self._current_worker = worker

                # Use Qt.DirectConnection so relay signals fire in the TrainingWorker
                # thread rather than being queued to GridSearchWorker's thread.
                # GridSearchWorker has no running event loop (it blocks on worker.wait()),
                # so queued signals to it would never be delivered — causing the epoch
                # bar to freeze and result_holder to stay empty.
                # With DirectConnection the relay runs in the emitter's thread, and from
                # there each GridSearchWorker signal is re-queued normally to the main
                # thread's event loop (which IS running), so the UI stays live.
                worker.log.connect(self.run_log, Qt.DirectConnection)
                worker.progress.connect(
                    lambda ep, tot, _i=i, _t=total: self.run_progress.emit(_i + 1, _t, ep, tot),
                    Qt.DirectConnection,
                )
                worker.epoch_metrics.connect(self.epoch_metrics, Qt.DirectConnection)

                worker.start()
                worker.wait()

                # Read the result directly from the worker object — TrainingWorker.run()
                # stores _result/_error on self before returning, so they are guaranteed
                # to be set by the time wait() returns (no event loop needed).
                if worker._result is not None:
                    res = worker._result
                elif worker._error is not None:
                    res = {"error": worker._error}
                else:
                    res = {"error": "No result"}
                results.append(res)

                # ── Save immediately in the worker thread (crash-safe) ──────
                # If the app crashes before all_done fires, each completed run
                # is already on disk and can be loaded via "Load Previous Runs".
                try:
                    from ..utils.results_saver import save_result
                    out_dir = self.config["training"].get("output_dir", "./results")
                    save_result(res, out_dir)
                except Exception:
                    import traceback as _tb
                    self.run_log.emit(
                        f"[WARNING] Could not save result for run {i+1} to disk:\n"
                        + _tb.format_exc(limit=3)
                    )

                self.run_finished.emit(i + 1, res)
                self._current_worker = None  # drop reference so the worker can be GC'd

            self.all_done.emit(results)

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())
