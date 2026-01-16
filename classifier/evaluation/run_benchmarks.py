# classifier/evaluation/run_benchmark.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

import numpy as np
import pandas as pd


# ----------------------------
# Helpers: split CSV handling
# ----------------------------

@dataclass
class SplitSpec:
    csv_path: Path
    has_fold: bool
    has_split: bool
    k: Optional[int]


def inspect_split_csv(csv_path: Path) -> SplitSpec:
    df = pd.read_csv(csv_path)

    has_fold = "fold" in df.columns
    has_split = "split" in df.columns

    if not has_fold and not has_split:
        raise ValueError(
            f"{csv_path} must contain either a 'split' column (train/test) "
            f"or a 'fold' column (0..k-1)."
        )

    k = int(df["fold"].nunique()) if has_fold else None

    # basic sanity checks
    if has_split:
        uniq = set(df["split"].astype(str).unique())
        if not uniq.issubset({"train", "test"}):
            raise ValueError(
                f"{csv_path}: 'split' column must contain only 'train' and 'test'. Found: {sorted(uniq)}"
            )

    if has_fold:
        # ensure folds are 0..k-1 (not required but helps)
        folds = sorted(df["fold"].unique().tolist())
        if folds and (min(folds) < 0):
            raise ValueError(f"{csv_path}: fold values must be >= 0. Found: {folds[:10]} ...")

    return SplitSpec(csv_path=csv_path, has_fold=has_fold, has_split=has_split, k=k)


def list_folds(spec: SplitSpec) -> List[Optional[int]]:
    # If there's no 'fold', we run one "fold" = None (single split mode)
    if not spec.has_fold:
        return [None]
    return list(range(spec.k or 0))


# ----------------------------
# Runner registry
# ----------------------------

RunnerFn = Callable[[Path, Optional[int], bool], Dict[str, Any]]
# signature:
#   run(split_csv: Path, fold: Optional[int], train: bool) -> Dict


def load_runners(selected: List[str]) -> Dict[str, RunnerFn]:
    """
    Each runner module should implement:
        run(split_csv: Path, fold: Optional[int], train: bool = True) -> Dict
    where fold=None means single split via 'split' column.
    """
    runners: Dict[str, RunnerFn] = {}

    if "ml" in selected:
        from classifier.evaluation.runners.ml_runner import run as run_ml
        runners["ml"] = run_ml

    if "resnet" in selected:
        from classifier.evaluation.runners.resnet50_runner import run as run_resnet
        runners["resnet"] = run_resnet

    if "vit" in selected:
        from classifier.evaluation.runners.vit_runner import run as run_vit
        runners["vit"] = run_vit

    missing = [m for m in selected if m not in runners]
    if missing:
        raise ValueError(
            f"Unknown/unsupported models: {missing}. Supported: ml,resnet,vit"
        )

    return runners


# ----------------------------
# Reporting
# ----------------------------

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def summarize(results: Dict[str, List[Dict[str, Any]]]) -> None:
    print("\n=== Summary ===")
    for model_name, per_fold in results.items():
        accs = [_safe_float(r.get("accuracy")) for r in per_fold]
        accs = [a for a in accs if a is not None]
        if not accs:
            print(f"{model_name}: no accuracy reported")
            continue

        accs = np.array(accs, dtype=float)
        if len(accs) == 1:
            print(f"{model_name}: accuracy={accs[0]:.4f}")
        else:
            # ddof=1 only valid if len>1
            std = accs.std(ddof=1) if len(accs) > 1 else 0.0
            print(
                f"{model_name}: accuracy mean={accs.mean():.4f} std={std:.4f} (k={len(accs)})"
            )


def results_to_df(results: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_name, per_fold in results.items():
        for r in per_fold:
            row = dict(r)
            row["model"] = model_name
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    # nice column ordering if present
    preferred = ["model", "fold", "accuracy", "loss", "train_time_sec", "eval_time_sec", "notes"]
    cols = preferred + [c for c in sorted(set(rows[0].keys())) if c not in preferred]
    df = pd.DataFrame(rows)
    # keep only cols that exist
    cols = [c for c in cols if c in df.columns]
    return df[cols]


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark multiple models on the SAME split/folds (supports k-fold via 'fold' column)."
    )
    ap.add_argument(
        "--split-csv",
        type=str,
        required=True,
        help="Path to split CSV (with split=train/test OR fold=0..k-1).",
    )
    ap.add_argument(
        "--models",
        type=str,
        default="ml,resnet,vit",
        help="Comma-separated list: ml,resnet,vit",
    )
    ap.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Run only one fold (requires 'fold' column). If omitted, runs all folds.",
    )
    ap.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training and only run evaluation (if runner supports it).",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional path to save per-fold results as CSV.",
    )
    args = ap.parse_args()

    split_csv = Path(args.split_csv).resolve()
    if not split_csv.exists():
        raise FileNotFoundError(split_csv)

    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    spec = inspect_split_csv(split_csv)

    runners = load_runners(selected)

    # Determine which folds to run
    folds = list_folds(spec)
    if args.fold is not None:
        if not spec.has_fold:
            raise ValueError("You passed --fold but the split CSV has no 'fold' column.")
        if args.fold not in folds:
            raise ValueError(f"Fold {args.fold} not in available folds: {folds}")
        folds = [args.fold]

    train_flag = not args.no_train

    all_results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in runners.keys()}

    for fold in folds:
        fold_label = f"fold={fold}" if fold is not None else "split=train/test"
        print(f"\n==============================")
        print(f"Running benchmark ({fold_label})")
        print(f"Split CSV: {split_csv}")
        print(f"Models: {', '.join(runners.keys())}")
        print(f"Train: {'yes' if train_flag else 'no'}")
        print(f"==============================")

        for model_name, runner in runners.items():
            print(f"\n--- {model_name.upper()} ---")
            out = runner(split_csv, fold)                        
            # out = runner(split_csv, fold, train_flag)


            # standardize fold in output for reporting
            out = dict(out)
            out.setdefault("fold", fold)

            if "accuracy" in out:
                print(f"{model_name} accuracy: {float(out['accuracy']):.4f}")
            else:
                print(f"{model_name}: no accuracy in output keys={list(out.keys())}")

            all_results[model_name].append(out)

    summarize(all_results)

    # Optional: save detailed results
    if args.out_csv:
        out_path = Path(args.out_csv).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = results_to_df(all_results)
        df.to_csv(out_path, index=False)
        print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
