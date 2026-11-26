"""Background runner: TabPFN on the LEAP authors' split, for every cut-off day and seed.

Reuses the cached features in extension_cache/, built by the notebook harness, and the same
train_test_split call as the authors' train_eval.run_benchmark. Each finished run is appended to
extension_results/tabpfn_per_window.csv straight away, and runs already in that file are skipped,
so starting the script again resumes where it stopped.

Usage, with the fyp environment:
    python extension/run_tabpfn.py --model tabpfn          # full training split as context
    python extension/run_tabpfn.py --model tabpfn_sub10k   # 10,000-row context per estimator
"""
import argparse
import logging
import sys
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

PROJECT = Path(__file__).resolve().parent.parent
REPO = PROJECT / "leap-early-prediction"
CACHE_DIR = PROJECT / "extension_cache"
RESULTS_DIR = PROJECT / "extension_results"
PER_WINDOW = RESULTS_DIR / "tabpfn_per_window.csv"
REFERENCE = REPO / "outputs" / "results_metrics_per_window.csv"
sys.path.insert(0, str(REPO))

from config import DEFAULT_RANDOM_STATE, DEFAULT_SEEDS, DEFAULT_TEST_SIZE  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from train_eval import _build_models  # noqa: E402

# Same setting as the notebook harness, so the cached features read back exactly as built.
pd.set_option("mode.string_storage", "python")

WINDOWS = [7, 14, 21, 28, 35, 42, 49, 56]
SEEDS = DEFAULT_SEEDS
METRIC_COLS = ["roc_auc", "f1", "pr_auc", "brier", "precision_at_10", "recall_at_10", "precision_at_20", "recall_at_20"]
ROW_COLS = ["t_days", "seed", "model", *METRIC_COLS, "fit_seconds", "predict_seconds"]
# Training rows each TabPFN estimator sees as context. None keeps the whole training split.
CONTEXT_ROWS = {"tabpfn": None, "tabpfn_sub10k": 10_000}

log = logging.getLogger("run_tabpfn")


@lru_cache(maxsize=None)
def load_xy(t):
    X_df = pd.read_parquet(CACHE_DIR / f"X_t{t}.parquet")
    y = pd.read_parquet(CACHE_DIR / f"y_t{t}.parquet")["y"]
    assert X_df.index.equals(y.index), f"t={t}: cached features and labels are not in the same row order"
    return X_df, y


def get_split(t, seed):
    """The authors' split: the same call as train_eval.run_benchmark, on the same row order."""
    X_df, y = load_xy(t)
    return train_test_split(X_df, y, test_size=DEFAULT_TEST_SIZE, random_state=seed, stratify=y)


def sanity_check():
    """GBDT at day 7, seed 0 must reproduce the authors' row exactly, as in the notebook."""
    X_train, X_test, y_train, y_test = get_split(7, 0)
    model = _build_models(["gbdt"], y_train)["gbdt"].fit(X_train, y_train)
    proba = pd.Series(model.predict_proba(X_test)[:, 1], index=y_test.index)
    mine = compute_metrics(y_test.reset_index(drop=True), proba.reset_index(drop=True))
    reference = pd.read_csv(REFERENCE)
    row = reference[(reference["model"] == "gbdt") & (reference["t_days"] == 7) & (reference["seed"] == 0)].iloc[0]
    worst = max(abs(mine[m] - float(row[m])) for m in METRIC_COLS)
    if not worst <= 1e-9:
        raise SystemExit(f"Sanity check FAILED, largest difference {worst:.3g}: the split or features differ.")
    log.info("Sanity check PASS: GBDT day 7, seed 0 matches the reproduction exactly")


def make_model(name, device):
    from tabpfn import TabPFNClassifier

    kwargs = {"device": device, "random_state": DEFAULT_RANDOM_STATE}
    if CONTEXT_ROWS[name] is not None:
        kwargs["inference_config"] = {"SUBSAMPLE_SAMPLES": CONTEXT_ROWS[name]}
    return TabPFNClassifier(**kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description="TabPFN on the LEAP authors' split, run in the background.")
    parser.add_argument("--model", choices=sorted(CONTEXT_ROWS), required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout, force=True
    )

    import tabpfn

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Model %s, tabpfn %s, device %s, writing to %s", args.model, tabpfn.__version__, device, PER_WINDOW)
    sanity_check()

    RESULTS_DIR.mkdir(exist_ok=True)
    done = set()
    if PER_WINDOW.exists():
        existing = pd.read_csv(PER_WINDOW)
        if list(existing.columns) != ROW_COLS:
            raise SystemExit(f"{PER_WINDOW.name} has columns {list(existing.columns)}, expected {ROW_COLS}")
        done = set(zip(existing["t_days"].astype(int), existing["seed"].astype(int), existing["model"].astype(str)))
    total = len(WINDOWS) * len(SEEDS)
    finished = sum((t, seed, args.model) in done for t in WINDOWS for seed in SEEDS)
    log.info("%d of %d runs already saved for %s", finished, total, args.model)

    for t in WINDOWS:
        for seed in SEEDS:
            if (t, seed, args.model) in done:
                continue
            X_train, X_test, y_train, y_test = get_split(t, seed)
            model = make_model(args.model, device)
            start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            proba_values = model.predict_proba(X_test)[:, 1]
            predict_seconds = time.perf_counter() - start
            classes = list(model.classes_)
            assert classes == [0, 1], f"classes are {classes}, so column 1 would not be y = 1"
            proba = pd.Series(proba_values, index=y_test.index)
            metrics = compute_metrics(y_test.reset_index(drop=True), proba.reset_index(drop=True))
            row = {"t_days": t, "seed": seed, "model": args.model, **metrics,
                   "fit_seconds": fit_seconds, "predict_seconds": predict_seconds}
            pd.DataFrame([row], columns=ROW_COLS).to_csv(PER_WINDOW, mode="a", header=not PER_WINDOW.exists(), index=False)
            finished += 1
            log.info(
                "%d/%d t=%d seed=%d %s: ROC-AUC %.4f, PR-AUC %.4f, Brier %.4f, fit %.1fs, predict %.1fs",
                finished, total, t, seed, args.model, metrics["roc_auc"], metrics["pr_auc"], metrics["brier"],
                fit_seconds, predict_seconds,
            )
    log.info("Finished: %d of %d runs saved for %s", finished, total, args.model)


if __name__ == "__main__":
    main()
