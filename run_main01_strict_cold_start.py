from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import main_01 as core
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Cannot import main_01.py. Put this file in the same directory as "
        "main_01.py, then run it again."
    ) from exc


OUTPUT_DIRECTORY_NAME = "Main01_Strict_Cold_Start_Output"


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


def entity_count(association: np.ndarray, mode: str) -> int:
    if mode == "disease":
        return int(association.shape[1])
    if mode == "mirna":
        return int(association.shape[0])
    raise ValueError(f"Unknown cold-start mode: {mode!r}")


def entity_degrees(association: np.ndarray, mode: str) -> np.ndarray:
    if mode == "disease":
        return np.asarray(association, dtype=np.int64).sum(axis=0).astype(np.int64)
    if mode == "mirna":
        return np.asarray(association, dtype=np.int64).sum(axis=1).astype(np.int64)
    raise ValueError(f"Unknown cold-start mode: {mode!r}")


def partition_entities_by_positive_load(
    association: np.ndarray,
    mode: str,
    entity_indices: np.ndarray,
    fold_count: int,
    seed: int,
) -> List[np.ndarray]:
    entities = np.asarray(entity_indices, dtype=np.int64).reshape(-1)
    if fold_count < 2:
        raise ValueError("fold_count must be at least two.")
    if len(entities) < fold_count:
        raise ValueError(
            f"Need at least {fold_count} evaluable entities, got {len(entities)}."
        )
    if len(np.unique(entities)) != len(entities):
        raise ValueError("Entity list contains duplicates.")

    degrees = entity_degrees(association, mode)[entities]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(entities))
    order = order[np.argsort(-degrees[order], kind="stable")]
    partitions: List[List[int]] = [[] for _ in range(fold_count)]
    positive_load = np.zeros(fold_count, dtype=np.int64)
    entity_load = np.zeros(fold_count, dtype=np.int64)
    for position in order:
        degree = int(degrees[position])
        candidates = np.flatnonzero(positive_load == positive_load.min())
        minimum_count = entity_load[candidates].min()
        candidates = candidates[entity_load[candidates] == minimum_count]
        target = int(rng.choice(candidates))
        partitions[target].append(int(entities[position]))
        positive_load[target] += degree
        entity_load[target] += 1

    result = [np.asarray(sorted(part), dtype=np.int64) for part in partitions]
    merged = np.concatenate(result)
    if set(merged.tolist()) != set(entities.tolist()) or len(merged) != len(entities):
        raise RuntimeError("Cold-start partition coverage audit failed.")
    return result


def make_cold_start_folds(
    association: np.ndarray, mode: str, fold_count: int, seed: int
):
    degrees = entity_degrees(association, mode)
    evaluable = np.flatnonzero(degrees > 0).astype(np.int64)
    test_partitions = partition_entities_by_positive_load(
        association, mode, evaluable, fold_count, seed
    )
    folds = []
    for test_entities in test_partitions:
        train_entities = evaluable[~np.isin(evaluable, test_entities)]
        if np.intersect1d(train_entities, test_entities).size:
            raise RuntimeError("Outer cold-start entity overlap detected.")
        folds.append((train_entities, test_entities))
    return folds


def make_inner_cold_start_split(
    association: np.ndarray,
    mode: str,
    outer_train_entities: np.ndarray,
    seed: int,
):
    inner_fold_count = max(
        2, int(round(1.0 / float(core.CONFIG.inner_validation_ratio)))
    )
    partitions = partition_entities_by_positive_load(
        association, mode, outer_train_entities, inner_fold_count, seed
    )
    validation_slot = int(seed % inner_fold_count)
    validation_entities = partitions[validation_slot]
    inner_entities = np.concatenate(
        [part for index, part in enumerate(partitions) if index != validation_slot]
    ).astype(np.int64)
    inner_entities.sort()
    if np.intersect1d(inner_entities, validation_entities).size:
        raise RuntimeError("Inner cold-start entity overlap detected.")
    return inner_entities, validation_entities


def sample_balanced_pairs_for_entities(
    association: np.ndarray, mode: str, entity_indices: np.ndarray, seed: int
):
    entities = np.asarray(entity_indices, dtype=np.int64).reshape(-1)
    if not len(entities):
        raise ValueError("Cannot sample pairs for an empty entity set.")
    rng = np.random.default_rng(seed)
    if mode == "disease":
        submatrix = np.asarray(association[:, entities], dtype=np.int64)
        positive_local = np.argwhere(submatrix > 0)
        unknown_local = np.argwhere(submatrix == 0)
        positive_mirna = positive_local[:, 0]
        positive_disease = entities[positive_local[:, 1]]
    elif mode == "mirna":
        submatrix = np.asarray(association[entities, :], dtype=np.int64)
        positive_local = np.argwhere(submatrix > 0)
        unknown_local = np.argwhere(submatrix == 0)
        positive_mirna = entities[positive_local[:, 0]]
        positive_disease = positive_local[:, 1]
    else:
        raise ValueError(f"Unknown cold-start mode: {mode!r}")

    positive_count = int(len(positive_local))
    if positive_count == 0:
        raise ValueError(f"The {mode} entity subset has no known positive edge.")
    if len(unknown_local) < positive_count:
        raise ValueError("Not enough unknown pairs for 1:1 negative sampling.")
    negative_local = unknown_local[
        rng.choice(len(unknown_local), size=positive_count, replace=False)
    ]
    if mode == "disease":
        negative_mirna = negative_local[:, 0]
        negative_disease = entities[negative_local[:, 1]]
    else:
        negative_mirna = entities[negative_local[:, 0]]
        negative_disease = negative_local[:, 1]

    mirna = np.concatenate([positive_mirna, negative_mirna]).astype(np.int64)
    disease = np.concatenate([positive_disease, negative_disease]).astype(np.int64)
    labels = np.concatenate(
        [
            np.ones(positive_count, dtype=np.int64),
            np.zeros(positive_count, dtype=np.int64),
        ]
    )
    order = rng.permutation(len(labels))
    mirna, disease, labels = mirna[order], disease[order], labels[order]
    used_entities = disease if mode == "disease" else mirna
    if not set(np.unique(used_entities)).issubset(set(entities.tolist())):
        raise RuntimeError("Pair sampler used an entity outside the requested subset.")
    if np.any(association[mirna[labels == 1], disease[labels == 1]] <= 0):
        raise RuntimeError("Positive-pair audit failed.")
    if np.any(association[mirna[labels == 0], disease[labels == 0]] != 0):
        raise RuntimeError("Unknown-pair audit failed.")
    return mirna, disease, labels


def audit_cold_start_arrays(
    mode: str,
    train_arrays,
    test_arrays,
    train_entities: np.ndarray,
    test_entities: np.ndarray,
) -> None:
    train_axis = np.asarray(train_arrays[1] if mode == "disease" else train_arrays[0])
    test_axis = np.asarray(test_arrays[1] if mode == "disease" else test_arrays[0])
    train_set = set(np.asarray(train_entities, dtype=np.int64).tolist())
    test_set = set(np.asarray(test_entities, dtype=np.int64).tolist())
    if train_set & test_set:
        raise RuntimeError("Train/test cold-start entity sets overlap.")
    if not set(np.unique(train_axis).tolist()).issubset(train_set):
        raise RuntimeError("Training pairs contain a held-out entity.")
    if not set(np.unique(test_axis).tolist()).issubset(test_set):
        raise RuntimeError("Test pairs contain a training entity.")
    if np.intersect1d(np.unique(train_axis), np.unique(test_axis)).size:
        raise RuntimeError("Strict cold-start audit found train/test entity overlap.")


def select_active_fusion_weights(labels, predictions, active):
    labels = np.asarray(labels, dtype=np.int64)
    active = np.asarray(active, dtype=np.bool_)
    ranked = [core.rank_normalize(value) for value in predictions]
    step = float(core.CONFIG.fusion_grid_step)
    units = int(round(1.0 / step))
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
        raise RuntimeError("Fusion search produced no candidate.")
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


def add_metrics(row: Dict[str, object], prefix: str, labels, probability) -> None:
    for metric, value in core.classification_metrics(labels, probability).items():
        row[f"{prefix}_{metric}"] = float(value)


def summarize_variant_rows(rows: Sequence[Mapping[str, object]]):
    variants = list(dict.fromkeys(str(row["variant"]) for row in rows))
    output = []
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        item: Dict[str, object] = {"variant": variant, "folds": len(selected)}
        for metric in core.METRIC_NAMES:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=0))
        output.append(item)
    full = next(row for row in output if row["variant"] == "full")
    for row in output:
        for metric in core.METRIC_NAMES:
            difference = float(full[f"{metric}_mean"]) - float(row[f"{metric}_mean"])
            row[f"full_minus_{metric}"] = difference
    return output


def write_entity_manifest(output_dir, data, mode, folds) -> None:
    names = data.disease_names if mode == "disease" else data.mirna_names
    degrees = entity_degrees(data.association, mode)
    rows = []
    for fold, (_, test_entities) in enumerate(folds, start=1):
        for entity in test_entities:
            rows.append(
                {
                    "fold": fold,
                    "held_out_entity_index": int(entity),
                    "held_out_entity_name": names[int(entity)],
                    "known_positive_degree": int(degrees[int(entity)]),
                }
            )
    core.write_csv(output_dir / "held_out_entities.csv", rows)
    zero_rows = [
        {
            "entity_index": int(entity),
            "entity_name": names[int(entity)],
            "reason": "zero known positives; no evaluable positive cold-start target",
        }
        for entity in np.flatnonzero(degrees == 0)
    ]
    core.write_csv(output_dir / "excluded_zero_degree_entities.csv", zero_rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_one_mode(data, mode: str, output_root: Path):
    mode_started = time.time()
    output_dir = output_root / f"{mode}_cold_start"
    output_dir.mkdir(parents=True, exist_ok=True)
    label = "disease" if mode == "disease" else "miRNA"
    core.progress("=" * 88)
    core.progress(f"Strict final-model {label} cold-start experiment started")
    offset = 0 if mode == "disease" else 100000
    folds = make_cold_start_folds(
        data.association, mode, core.CONFIG.folds, core.CONFIG.seed + offset
    )
    write_entity_manifest(output_dir, data, mode, folds)

    fold_rows: List[Dict[str, object]] = []
    variant_rows: List[Dict[str, object]] = []
    for fold, (outer_train_entities, test_entities) in enumerate(folds, start=1):
        fold_started = time.time()
        fold_seed = core.CONFIG.seed + fold * 10007 + (0 if mode == "disease" else 200000)
        core.progress("-" * 88)
        core.progress(
            f"{label} cold-start fold {fold}/{core.CONFIG.folds}: "
            f"train entities={len(outer_train_entities)}, test entities={len(test_entities)}"
        )
        train_arrays = sample_balanced_pairs_for_entities(
            data.association, mode, outer_train_entities, fold_seed + 101
        )
        test_arrays = sample_balanced_pairs_for_entities(
            data.association, mode, test_entities, fold_seed + 103
        )
        audit_cold_start_arrays(
            mode, train_arrays, test_arrays, outer_train_entities, test_entities
        )

        inner_entities, validation_entities = make_inner_cold_start_split(
            data.association, mode, outer_train_entities, fold_seed + 107
        )
        inner_arrays = sample_balanced_pairs_for_entities(
            data.association, mode, inner_entities, fold_seed + 109
        )
        validation_arrays = sample_balanced_pairs_for_entities(
            data.association, mode, validation_entities, fold_seed + 113
        )
        audit_cold_start_arrays(
            mode, inner_arrays, validation_arrays, inner_entities, validation_entities
        )
        if np.intersect1d(validation_entities, test_entities).size:
            raise RuntimeError("Inner validation contains an outer-test entity.")

        core.progress(f"Fold {fold}: selecting branch-A epoch on unseen validation entities")
        a_epoch, a_inner_auc, _, validation_a = core.select_bio_epoch(
            data, inner_arrays, validation_arrays, fold_seed + 4011
        )

        core.progress(f"Fold {fold}: fitting branch B on inner cold-start training entities")
        inner_stage = core.build_stage_data(
            data.association.shape, inner_arrays, fold_seed + 4513
        )
        inner_b = core.fit_structural_expert(
            inner_stage, inner_arrays, data, fold_seed + 4519
        )
        validation_b = core.predict_structural_expert(
            inner_b, inner_stage, validation_arrays, data
        )
        b_inner_auc = core.safe_auc(validation_arrays[2], validation_b)
        del inner_stage, inner_b

        core.progress(f"Fold {fold}: selecting branch-C epoch on unseen validation entities")
        c_epoch, c_inner_auc, _, validation_c = core.select_completion_epoch(
            data, inner_arrays, validation_arrays, fold_seed + 4731
        )

        validation_predictions = (validation_a, validation_b, validation_c)
        full_weights, full_inner_auc, full_inner_aupr, _ = select_active_fusion_weights(
            validation_arrays[2], validation_predictions, (True, True, True)
        )
        without_a_weights, without_a_inner_auc, without_a_inner_aupr, _ = (
            select_active_fusion_weights(
                validation_arrays[2], validation_predictions, (False, True, True)
            )
        )
        core.progress(
            f"Fold {fold}: full weights={tuple(float(x) for x in full_weights)}; "
            f"w/o-A weights={tuple(float(x) for x in without_a_weights)}"
        )

        core.progress(f"Fold {fold}: refitting branch A for {a_epoch} epoch(s)")
        a_model, a_ema, _ = core.fit_bio_from_scratch(
            data, train_arrays, a_epoch, fold_seed + 5017
        )
        test_a = core.predict_bio(a_model, test_arrays, a_ema)
        del a_model, a_ema

        core.progress(f"Fold {fold}: refitting branch B")
        b_stage = core.build_stage_data(
            data.association.shape, train_arrays, fold_seed + 6023
        )
        b_model = core.fit_structural_expert(
            b_stage, train_arrays, data, fold_seed + 6029
        )
        test_b = core.predict_structural_expert(b_model, b_stage, test_arrays, data)
        del b_stage, b_model

        core.progress(f"Fold {fold}: refitting branch C for {c_epoch} epoch(s)")
        c_model, _ = core.fit_completion_from_scratch(
            data, train_arrays, c_epoch, fold_seed + 7037
        )
        test_c = core.predict_completion(c_model, test_arrays)
        del c_model

        test_predictions = (test_a, test_b, test_c)
        full_probability = fuse_test_predictions(test_predictions, full_weights)
        without_a_probability = fuse_test_predictions(
            test_predictions, without_a_weights
        )
        labels = np.asarray(test_arrays[2], dtype=np.int64)
        full_metrics = core.classification_metrics(labels, full_probability)
        without_a_metrics = core.classification_metrics(labels, without_a_probability)

        row: Dict[str, object] = {
            "mode": mode,
            "fold": fold,
            "train_entity_count": len(outer_train_entities),
            "test_entity_count": len(test_entities),
            "train_pair_count": len(train_arrays[2]),
            "test_pair_count": len(test_arrays[2]),
            "A_selected_epoch": int(a_epoch),
            "C_selected_epoch": int(c_epoch),
            "A_inner_AUC": float(a_inner_auc),
            "B_inner_AUC": float(b_inner_auc),
            "C_inner_AUC": float(c_inner_auc),
            "full_inner_AUC": float(full_inner_auc),
            "full_inner_AUPR": float(full_inner_aupr),
            "full_A_weight": float(full_weights[0]),
            "full_B_weight": float(full_weights[1]),
            "full_C_weight": float(full_weights[2]),
            "without_A_inner_AUC": float(without_a_inner_auc),
            "without_A_inner_AUPR": float(without_a_inner_aupr),
            "without_A_B_weight": float(without_a_weights[1]),
            "without_A_C_weight": float(without_a_weights[2]),
        }
        add_metrics(row, "FULL", labels, full_probability)
        add_metrics(row, "WITHOUT_A", labels, without_a_probability)
        add_metrics(row, "BRANCH_A", labels, test_a)
        add_metrics(row, "BRANCH_B", labels, test_b)
        add_metrics(row, "BRANCH_C", labels, test_c)
        fold_rows.append(row)

        probabilities = {
            "full": full_probability,
            "without_branch_A": without_a_probability,
            "branch_A_only": test_a,
            "branch_B_only": test_b,
            "branch_C_only": test_c,
        }
        for variant, probability in probabilities.items():
            variant_rows.append(
                {
                    "mode": mode,
                    "fold": fold,
                    "variant": variant,
                    **core.classification_metrics(labels, probability),
                }
            )

        np.savez_compressed(
            output_dir / f"fold_{fold}_predictions.npz",
            labels=labels,
            mirna_indices=np.asarray(test_arrays[0], dtype=np.int64),
            disease_indices=np.asarray(test_arrays[1], dtype=np.int64),
            test_entities=np.asarray(test_entities, dtype=np.int64),
            branch_A=np.asarray(test_a, dtype=np.float32),
            branch_B=np.asarray(test_b, dtype=np.float32),
            branch_C=np.asarray(test_c, dtype=np.float32),
            full_probability=np.asarray(full_probability, dtype=np.float32),
            without_A_probability=np.asarray(without_a_probability, dtype=np.float32),
            full_weights=np.asarray(full_weights, dtype=np.float32),
            without_A_weights=np.asarray(without_a_weights, dtype=np.float32),
        )
        core.write_csv(output_dir / "all_fold_metrics.csv", fold_rows)
        core.write_csv(output_dir / "variant_fold_metrics.csv", variant_rows)
        core.write_csv(
            output_dir / "cold_start_summary.csv",
            summarize_variant_rows(variant_rows),
        )
        core.progress(
            f"Fold {fold} full: AUC={full_metrics['AUC']:.6f}, "
            f"AUPR={full_metrics['AUPR']:.6f}; w/o A: "
            f"AUC={without_a_metrics['AUC']:.6f}, AUPR={without_a_metrics['AUPR']:.6f}"
        )
        core.progress(
            f"Fold {fold} completed in {core.format_duration(time.time() - fold_started)}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_variant_rows(variant_rows)
    core.write_csv(output_dir / "all_fold_metrics.csv", fold_rows)
    core.write_csv(output_dir / "variant_fold_metrics.csv", variant_rows)
    core.write_csv(output_dir / "cold_start_summary.csv", summary)
    core.progress(
        f"{label} cold-start completed in {core.format_duration(time.time() - mode_started)}"
    )
    return summary


def configure_project_root(project_root: Path | None) -> None:
    if project_root is None:
        return
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


def parse_modes(values: Sequence[str]) -> List[str]:
    modes: List[str] = []
    for value in values:
        modes.extend(part.strip().lower() for part in value.split(",") if part.strip())
    if not modes or "both" in modes or "all" in modes:
        return ["disease", "mirna"]
    aliases = {"disease": "disease", "mirna": "mirna"}
    unknown = [mode for mode in modes if mode not in aliases]
    if unknown:
        raise ValueError(f"Unknown cold-start mode(s): {unknown}")
    return list(dict.fromkeys(aliases[mode] for mode in modes))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict cold-start runner for final main_01.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["both"],
        help="disease, mirna, or both (space/comma separated).",
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_DIRECTORY_NAME))
    parser.add_argument(
        "--quick-check",
        action="store_true",
        help="Two-fold one-epoch smoke test only; metrics are not for the paper.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_project_root(args.project_root)
    verify_final_model_contract()
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
    modes = parse_modes(args.modes)
    output_root = args.output_dir.expanduser()
    if not output_root.is_absolute():
        output_root = core.SERVER_ROOT / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_path = Path(core.__file__).resolve()
    manifest = {
        "runner": Path(__file__).name,
        "model_source": str(source_path),
        "model_source_sha256": file_sha256(source_path),
        "model_version": "main_01 final compact B/C model",
        "modes": modes,
        "config": asdict(core.CONFIG),
        "split_unit": "whole entity on the selected cold-start axis",
        "inner_validation": "entity-disjoint from inner training and outer test",
        "negative_sampling": "all positives plus an equal number of unknown pairs per entity subset",
        "graph_leakage_control": (
            "B/C association matrices are reconstructed only from positive pairs "
            "whose cold-start entities belong to the corresponding training subset."
        ),
        "fusion": (
            "complete non-negative simplex with no branch-specific bounds; "
            "selected using entity-disjoint inner validation data only"
        ),
    }
    (output_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    core.set_seed(core.CONFIG.seed)
    data = core.load_model_data()
    combined = []
    for mode in modes:
        for row in run_one_mode(data, mode, output_root):
            combined.append({"mode": mode, **row})
        core.write_csv(output_root / "combined_cold_start_summary.csv", combined)
    core.progress(f"All strict cold-start outputs saved to {output_root}")


if __name__ == "__main__":
    main()

