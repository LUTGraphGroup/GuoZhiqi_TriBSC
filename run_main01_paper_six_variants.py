from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import main_01 as core
except ModuleNotFoundError as exc:
    if exc.name == "main_01":
        raise SystemExit(
            "Cannot find main_01.py. Put this runner beside main_01.py and run it again."
        ) from exc
    raise


OUTPUT_DIRECTORY_NAME = "Main01_Paper_SixVariants_Output"

BRANCH_VARIANTS: Dict[str, Tuple[bool, bool, bool]] = {
    "branch_A_only": (True, False, False),
    "branch_B_only": (False, True, False),
    "branch_C_only": (False, False, True),
    "without_branch_A": (False, True, True),
    "without_branch_B": (True, False, True),
    "without_branch_C": (True, True, False),
}

PAPER_LABELS: Dict[str, str] = {
    "full": "TriBSC",
    "branch_A_only": "Biological-only",
    "branch_B_only": "Structural-only",
    "branch_C_only": "Completion-only",
    "without_branch_A": "w/o Biological",
    "without_branch_B": "w/o Structural",
    "without_branch_C": "w/o Completion",
}


def verify_final_model_contract() -> None:
    expected_b = (
        "path3_normalized", "path3_raw", "path3_resource_allocation",
        "static_left", "static_right", "gip_left", "gip_both",
        "mirna_degree", "disease_degree",
    )
    expected_c_propagation = (
        "static_left", "static_right", "functional_left",
        "functional_semantic_both",
    )
    expected_c_auxiliary = (
        "mirna_degree", "disease_degree", "degree_product",
    )
    if tuple(core.STRUCTURAL_FEATURE_NAMES) != expected_b:
        raise RuntimeError(
            "The imported main_01.py is not the final 9-feature branch-B model."
        )
    if tuple(core.COMPLETION_PROPAGATION_NAMES) != expected_c_propagation:
        raise RuntimeError(
            "The imported main_01.py is not the final 4-propagation-channel branch-C model."
        )
    if tuple(core.COMPLETION_AUXILIARY_NAMES) != expected_c_auxiliary:
        raise RuntimeError(
            "The imported main_01.py has an unexpected branch-C auxiliary schema."
        )


def build_catalog() -> List[Dict[str, object]]:
    variants = {"full": (True, True, True), **BRANCH_VARIANTS}
    return [
        {
            "variant": name,
            "paper_label": PAPER_LABELS[name],
            "category": (
                "baseline" if name == "full"
                else "branch_ablation" if name.startswith("without_")
                else "single_branch"
            ),
            "active_branches": "".join(
                letter for letter, keep in zip("ABC", mask) if keep
            ),
            "description": (
                "Full TriBSC reference with unchanged branches A, B, and C."
                if name == "full"
                else "Retain only the listed branches; select fusion weights "
                     "on the inner validation set using those branches only."
            ),
        }
        for name, mask in variants.items()
    ]


def select_active_fusion_weights(
    labels: np.ndarray,
    predictions: Sequence[np.ndarray],
    active: Sequence[bool],
):
    labels = np.asarray(labels, dtype=np.int64)
    active = np.asarray(active, dtype=np.bool_)
    if active.shape != (3,) or not active.any():
        raise ValueError(f"Expected a nonempty three-branch mask, got {active}.")
    ranked = [core.rank_normalize(value) for value in predictions]
    step = float(core.CONFIG.fusion_grid_step)
    units = int(round(1.0 / step))
    if units < 1 or not np.isclose(units * step, 1.0):
        raise ValueError("fusion_grid_step must divide one exactly.")

    neutral = active.astype(np.float64) / float(active.sum())
    best_key = None
    best_weight = None
    best_probability = None
    for a_unit in range(units + 1):
        for b_unit in range(units - a_unit + 1):
            c_unit = units - a_unit - b_unit
            weight = np.asarray([a_unit, b_unit, c_unit], dtype=np.float64) / units
            if np.any(weight[~active] != 0.0):
                continue
            probability = sum(float(w) * p for w, p in zip(weight, ranked))
            auc = core.safe_auc(labels, probability)
            aupr = float(average_precision_score(labels, probability))
            distance = float(np.square(weight - neutral).sum())
            key = (round(auc, 10), round(aupr, 10), -distance)
            if best_key is None or key > best_key:
                best_key = key
                best_weight = weight
                best_probability = probability
    if best_weight is None or best_probability is None:
        raise RuntimeError("No valid fusion-weight candidate was generated.")
    return (
        best_weight.astype(np.float32),
        float(best_key[0]),
        float(best_key[1]),
        np.asarray(best_probability, dtype=np.float32),
    )


def fuse_test_predictions(predictions, weights):
    ranked = [core.rank_normalize(value) for value in predictions]
    return np.asarray(
        sum(float(w) * p for w, p in zip(weights, ranked)), dtype=np.float32
    )


def metric_summary(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    variants = list(dict.fromkeys(str(row["variant"]) for row in rows))
    output: List[Dict[str, object]] = []
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        item: Dict[str, object] = {
            "variant": variant,
            "paper_label": selected[0]["paper_label"],
            "category": selected[0]["category"],
            "active_branches": selected[0]["active_branches"],
            "description": selected[0]["description"],
            "folds": len(selected),
        }
        for metric in core.METRIC_NAMES:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=0))
        output.append(item)

    baseline = next(row for row in output if row["variant"] == "full")
    for row in output:
        for metric in core.METRIC_NAMES:
            delta = float(row[f"{metric}_mean"]) - float(baseline[f"{metric}_mean"])
            row[f"delta_{metric}_vs_full"] = delta
            row[f"full_minus_{metric}"] = -delta
    return output


def configure_project_root(project_root: Path) -> None:
    root = project_root.expanduser().resolve()
    core.SERVER_ROOT = root
    core.D14_DIR = Path(
        core.os.environ.get("MDA_D14_CACHE_DIR", root / "D14_UnifiedDataCache")
    )
    core.FOUNDATION_DIR = Path(
        core.os.environ.get("MDA_FOUNDATION_CACHE_DIR", root / "Bio_Foundation_Cache")
    )
    core.EXTERNAL_CACHE_DIR = Path(
        core.os.environ.get("MDA_EXTERNAL_CACHE_DIR", root / "D16_BioHeteroPath_Cache")
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_ablation_experiment(output_dir: Path, quick_check: bool = False):
    started = time.time()
    verify_final_model_contract()
    output_dir.mkdir(parents=True, exist_ok=True)
    core.progress("=" * 88)
    core.progress("Paper branch-level experiment: full TriBSC + six variants")
    core.progress(f"Device: {core.DEVICE}; output: {output_dir}")
    if quick_check:
        core.progress("QUICK CHECK ONLY: these metrics must not be used in the paper.")
    core.set_seed(core.CONFIG.seed)
    data = core.load_model_data()
    all_m, all_d, all_y = core.sample_random_balanced_pairs(
        data.association, core.CONFIG.seed
    )
    folds = core.make_balanced_folds(all_y, core.CONFIG.folds, core.CONFIG.seed)

    variants = {"full": (True, True, True), **BRANCH_VARIANTS}
    catalog = build_catalog()
    catalog_by_name = {str(row["variant"]): row for row in catalog}
    core.write_csv(output_dir / "ablation_catalog.csv", catalog)
    manifest = {
        "runner": Path(__file__).name,
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "model_source": str(Path(core.__file__).resolve()),
        "model_source_sha256": file_sha256(Path(core.__file__).resolve()),
        "model_version": "main_01 final compact B/C model",
        "config": asdict(core.CONFIG),
        "quick_check": bool(quick_check),
        "paper_variant_count": len(BRANCH_VARIANTS),
        "includes_full_model_reference": True,
        "branch_variants": variants,
        "paper_labels": PAPER_LABELS,
        "branch_A_policy": "Use the original biological branch without component ablations.",
        "control_policy": (
            "Identical sampled pairs, outer folds, and inner splits for all variants. "
            "Each unchanged branch is trained once per fold and its predictions are reused. "
            "Fusion weights are selected on inner validation data using retained branches only; "
            "excluded branches have zero fusion weight. The outer test set is not used "
            "for selecting epochs or fusion weights."
        ),
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    all_rows: List[Dict[str, object]] = []

    for fold, (train_index, test_index) in enumerate(folds, start=1):
        fold_started = time.time()
        core.progress("-" * 88)
        core.progress(f"Ablation fold {fold}/{core.CONFIG.folds} started")
        train_arrays = (all_m[train_index], all_d[train_index], all_y[train_index])
        test_arrays = (all_m[test_index], all_d[test_index], all_y[test_index])
        inner_index, validation_index = core.make_inner_split(
            train_arrays[2], core.CONFIG.seed + fold * 1009
        )
        inner_arrays = tuple(value[inner_index] for value in train_arrays)
        validation_arrays = tuple(value[validation_index] for value in train_arrays)

        # Preserve the original runner's branch training calls and random seeds.
        core.progress(f"Fold {fold}: preparing shared branch-B validation prediction")
        inner_stage = core.build_stage_data(
            data.association.shape, inner_arrays, core.CONFIG.seed + fold * 4507 + 13
        )
        inner_expert = core.fit_structural_expert(
            inner_stage, inner_arrays, data, core.CONFIG.seed + fold * 4507 + 19
        )
        validation_b = core.predict_structural_expert(
            inner_expert, inner_stage, validation_arrays, data
        )
        b_inner_auc = core.safe_auc(validation_arrays[2], validation_b)
        del inner_stage, inner_expert

        core.progress(f"Fold {fold}: selecting shared branch-C epoch")
        c_epoch, c_inner_auc, _, validation_c = core.select_completion_epoch(
            data, inner_arrays, validation_arrays, core.CONFIG.seed + fold * 4703 + 31
        )

        core.progress(f"Fold {fold}: refitting shared branch B")
        b_stage = core.build_stage_data(
            data.association.shape, train_arrays, core.CONFIG.seed + fold * 6007 + 23
        )
        b_model = core.fit_structural_expert(
            b_stage, train_arrays, data, core.CONFIG.seed + fold * 6007 + 29
        )
        test_b = core.predict_structural_expert(b_model, b_stage, test_arrays, data)
        del b_stage, b_model

        core.progress(f"Fold {fold}: refitting shared branch C for {c_epoch} epoch(s)")
        c_model, _ = core.fit_completion_from_scratch(
            data, train_arrays, c_epoch, core.CONFIG.seed + fold * 7001 + 37
        )
        test_c = core.predict_completion(c_model, test_arrays)
        del c_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        core.progress(f"Fold {fold}: selecting the unchanged branch-A epoch")
        a_epoch, a_inner_auc, _, validation_a = core.select_bio_epoch(
            data, inner_arrays, validation_arrays, core.CONFIG.seed + fold * 4001 + 11
        )
        core.progress(f"Fold {fold}: refitting unchanged branch A for {a_epoch} epoch(s)")
        a_model, a_ema, _ = core.fit_bio_from_scratch(
            data, train_arrays, a_epoch, core.CONFIG.seed + fold * 5003 + 17
        )
        test_a = core.predict_bio(a_model, test_arrays, a_ema)
        del a_model, a_ema
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        validation_predictions = (validation_a, validation_b, validation_c)
        test_predictions = (test_a, test_b, test_c)
        fold_predictions: Dict[str, np.ndarray] = {
            "labels": np.asarray(test_arrays[2], dtype=np.int64),
            "mirna_indices": np.asarray(test_arrays[0], dtype=np.int64),
            "disease_indices": np.asarray(test_arrays[1], dtype=np.int64),
            "branch_A": np.asarray(test_a, dtype=np.float32),
            "branch_B": np.asarray(test_b, dtype=np.float32),
            "branch_C": np.asarray(test_c, dtype=np.float32),
        }

        for name, active in variants.items():
            weights, inner_auc, inner_aupr, _ = select_active_fusion_weights(
                validation_arrays[2], validation_predictions, active
            )
            final_probability = fuse_test_predictions(test_predictions, weights)
            metrics = core.classification_metrics(test_arrays[2], final_probability)
            all_rows.append({
                "fold": fold,
                **catalog_by_name[name],
                "A_selected_epoch": int(a_epoch) if active[0] else "",
                "C_selected_epoch": int(c_epoch) if active[2] else "",
                "A_inner_AUC": float(a_inner_auc),
                "B_inner_AUC": float(b_inner_auc),
                "C_inner_AUC": float(c_inner_auc),
                "fusion_inner_AUC": float(inner_auc),
                "fusion_inner_AUPR": float(inner_aupr),
                "fusion_A_weight": float(weights[0]),
                "fusion_B_weight": float(weights[1]),
                "fusion_C_weight": float(weights[2]),
                **metrics,
            })
            fold_predictions[f"variant_{name}"] = final_probability
            fold_predictions[f"variant_{name}_weights"] = weights
            core.progress(
                f"Fold {fold} {PAPER_LABELS[name]}: AUC={metrics['AUC']:.6f}, "
                f"AUPR={metrics['AUPR']:.6f}, weights={tuple(float(x) for x in weights)}"
            )

        np.savez_compressed(output_dir / f"fold_{fold}_predictions.npz", **fold_predictions)
        core.write_csv(output_dir / "all_fold_metrics.csv", all_rows)
        core.write_csv(output_dir / "all_ablation_summary.csv", metric_summary(all_rows))
        core.progress(
            f"Ablation fold {fold} completed in "
            f"{core.format_duration(time.time() - fold_started)}"
        )

    summary = metric_summary(all_rows)
    core.write_csv(output_dir / "all_fold_metrics.csv", all_rows)
    core.write_csv(output_dir / "all_ablation_summary.csv", summary)
    core.progress(f"All ablation outputs saved to {output_dir}")
    core.progress(
        f"Ablation experiment completed in {core.format_duration(time.time() - started)}"
    )
    core.progress("=" * 88)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the paper's six branch-level variants and the full TriBSC reference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--list-ablations", action="store_true",
        help="List the full-model reference and the six paper variants, then exit.",
    )
    parser.add_argument(
        "--project-root", type=Path, default=SCRIPT_DIR,
        help="Directory containing the data and cache folders.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_DIRECTORY_NAME))
    parser.add_argument(
        "--quick-check", action="store_true",
        help="Two-fold one-epoch smoke test only; metrics are not for the paper.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.list_ablations:
        for entry in build_catalog():
            print(
                f"{entry['variant']:24s} {entry['paper_label']:20s} "
                f"active={entry['active_branches']}"
            )
        return

    configure_project_root(args.project_root)
    if args.quick_check:
        core.CONFIG = replace(
            core.CONFIG,
            folds=2,
            crossfit_graph_views=2,
            maximum_epochs=1,
            patience=1,
            expert_max_iter=5,
            completion_maximum_epochs=1,
            completion_patience=1,
        )
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = core.SERVER_ROOT / output_dir
    run_ablation_experiment(output_dir.resolve(), quick_check=args.quick_check)


if __name__ == "__main__":
    main()
