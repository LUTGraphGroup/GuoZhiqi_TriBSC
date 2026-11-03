from __future__ import annotations

import argparse
import csv
import difflib
import importlib.util
import json
import math
import os
import pickle
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

try:
    import pandas as pd
except Exception:  # pandas is optional unless HMDD/xlsx output is requested
    pd = None

def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def safe_name(text: str, max_len: int = 80) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return (value[:max_len] or "item").strip("_") or "item"


def norm_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def canonical_mirna(value: Any) -> str:
    text = norm_text(value)
    text = text.replace("_", "-").replace(" ", "")
    # Species prefix is frequently present in HMDD but absent in benchmark name lists.
    for prefix in ("hsa-", "human-"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.replace("mirna", "mir")
    return text


def canonical_disease(value: Any) -> str:
    text = norm_text(value)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    replacements = {
        "neoplasms": "cancer",
        "neoplasm": "cancer",
        "tumours": "cancer",
        "tumour": "cancer",
        "tumors": "cancer",
        "tumor": "cancer",
    }
    tokens = [replacements.get(token, token) for token in text.split()]
    # Avoid changing e.g. "colonic" into "colon" automatically: that is an alias,
    # not a normalization. Users can provide --disease_alias when needed.
    return " ".join(tokens)


def parse_aliases(items: Sequence[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --disease_alias {item!r}; expected 'MODEL_NAME=DATABASE_NAME'."
            )
        left, right = item.split("=", 1)
        left, right = left.strip(), right.strip()
        if not left or not right:
            raise ValueError(f"Invalid --disease_alias {item!r}.")
        aliases[canonical_disease(left)] = canonical_disease(right)
    return aliases




def log_combination(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


def hypergeometric_survival(k_observed: int, population: int, successes: int, draws: int) -> float:
    population, successes, draws, k_observed = map(int, (population, successes, draws, k_observed))
    if population <= 0 or draws < 0 or successes < 0 or successes > population or draws > population:
        return float("nan")
    lower = max(k_observed, 0, draws - (population - successes))
    upper = min(draws, successes)
    if lower > upper:
        return 0.0
    denominator = log_combination(population, draws)
    logs = [
        log_combination(successes, x)
        + log_combination(population - successes, draws - x)
        - denominator
        for x in range(lower, upper + 1)
    ]
    maximum = max(logs)
    return float(min(1.0, math.exp(maximum) * sum(math.exp(v - maximum) for v in logs)))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        # Still create a useful empty file.
        path.write_text("", encoding="utf-8-sig")
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def maybe_write_excel(path: Path, sheets: Mapping[str, Sequence[Mapping[str, Any]]]) -> bool:
    if pd is None:
        log("pandas is unavailable; skipping combined Excel workbook (CSV/JSON outputs remain complete).")
        return False
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            used: Set[str] = set()
            for raw_name, rows in sheets.items():
                name = re.sub(r"[\\/*?:\[\]]+", "_", raw_name)[:31] or "Sheet"
                base = name
                counter = 2
                while name in used:
                    suffix = f"_{counter}"
                    name = (base[: 31 - len(suffix)] + suffix)[:31]
                    counter += 1
                used.add(name)
                pd.DataFrame(list(rows)).to_excel(writer, sheet_name=name, index=False)
        return True
    except Exception as exc:
        log(f"Excel export skipped because it failed: {exc}")
        return False

def configure_model_environment(args: argparse.Namespace) -> None:
    mapping = {
        "MDA_DATA_DIR": args.data_dir,
        "MDA_D14_CACHE_DIR": args.d14_cache_dir,
        "MDA_FOUNDATION_CACHE_DIR": args.foundation_cache_dir,
        "MDA_EXTERNAL_CACHE_DIR": args.external_cache_dir,
    }
    for key, value in mapping.items():
        if value:
            os.environ[key] = str(Path(value).expanduser().resolve())


def load_model_module(model_path: Path):
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    module_name = "strict_three_branch_user_model"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    required = [
        "CONFIG", "DEVICE", "load_model_data", "make_inner_split",
        "select_bio_epoch", "fit_bio_from_scratch", "predict_bio",
        "build_stage_data", "fit_structural_expert", "predict_structural_expert",
        "select_completion_epoch", "fit_completion_from_scratch", "predict_completion",
        "select_fusion_weights", "rank_fuse_three", "set_seed",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            "The model file is not compatible with this case-study script. Missing: "
            + ", ".join(missing)
        )
    return module

@dataclass
class HMDDEvidence:
    pmids: Set[str]
    record_count: int = 0


def load_hmdd_pairs(path: Optional[Path]) -> Dict[Tuple[str, str], HMDDEvidence]:
    if path is None:
        return {}
    if pd is None:
        raise RuntimeError("pandas + openpyxl are required when --hmdd_xlsx is used.")
    frame = pd.read_excel(path)
    lower = {str(column).strip().lower(): column for column in frame.columns}
    if "mirna" not in lower or "disease" not in lower:
        raise ValueError(
            f"HMDD file must contain miRNA and disease columns. Got: {list(frame.columns)}"
        )
    mir_col, dis_col = lower["mirna"], lower["disease"]
    pmid_col = lower.get("pmid")
    result: Dict[Tuple[str, str], HMDDEvidence] = {}
    for _, row in frame.iterrows():
        mir = canonical_mirna(row[mir_col])
        disease = canonical_disease(row[dis_col])
        if not mir or not disease or mir == "nan" or disease == "nan":
            continue
        key = (mir, disease)
        if key not in result:
            result[key] = HMDDEvidence(pmids=set(), record_count=0)
        result[key].record_count += 1
        if pmid_col is not None:
            pmid_value = row[pmid_col]
            if not pd.isna(pmid_value):
                text = str(pmid_value).strip()
                if text.endswith(".0") and text[:-2].isdigit():
                    text = text[:-2]
                if text:
                    result[key].pmids.add(text)
    return result


def hmdd_evidence_for(
    hmdd: Mapping[Tuple[str, str], HMDDEvidence],
    mirna_name: str,
    disease_name: str,
    aliases: Mapping[str, str],
) -> Dict[str, Any]:
    if not hmdd:
        return {
            "HMDD_Corroborated": "",
            "HMDD_RecordCount": "",
            "HMDD_PMID_Count": "",
            "HMDD_PMIDs": "",
        }
    model_key = canonical_disease(disease_name)
    disease_key = aliases.get(model_key, model_key)
    item = hmdd.get((canonical_mirna(mirna_name), disease_key))
    if item is None:
        return {
            "HMDD_Corroborated": "NO",
            "HMDD_RecordCount": 0,
            "HMDD_PMID_Count": 0,
            "HMDD_PMIDs": "",
        }
    pmids = sorted(item.pmids)
    return {
        "HMDD_Corroborated": "YES",
        "HMDD_RecordCount": item.record_count,
        "HMDD_PMID_Count": len(pmids),
        "HMDD_PMIDs": ";".join(pmids),
    }


def load_dbdemc_cancer_types(path: Optional[Path]) -> Set[str]:
    if path is None:
        return set()
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"dbDEMC file has no header: {path}")
        lookup = {name.strip().lower(): name for name in reader.fieldnames}
        if "cancertype" not in lookup:
            raise ValueError(
                f"dbDEMC dataset-list file must contain CancerType. Got: {reader.fieldnames}"
            )
        column = lookup["cancertype"]
        types: Set[str] = set()
        for row in reader:
            value = row.get(column, "") or ""
            for part in str(value).split(","):
                current = canonical_disease(part)
                if current:
                    types.add(current)
        return types


def dbdemc_disease_coverage(
    disease_name: str,
    cancer_types: Set[str],
    aliases: Mapping[str, str],
) -> bool:
    if not cancer_types:
        return False
    key = canonical_disease(disease_name)
    key = aliases.get(key, key)
    # Disease-level metadata only. Substring fallback is intentionally used here
    # because this result is NOT assigned as miRNA-pair evidence.
    return any(key == item or key in item or item in key for item in cancer_types)

def resolve_target_indices(disease_names: Sequence[str], targets: Sequence[str]) -> Dict[str, int]:
    normalized = {norm_text(name): index for index, name in enumerate(disease_names)}
    result: Dict[str, int] = {}
    for requested in targets:
        key = norm_text(requested)
        if key in normalized:
            result[disease_names[normalized[key]]] = normalized[key]
            continue

        contains = [
            (index, name) for index, name in enumerate(disease_names)
            if key in norm_text(name) or norm_text(name) in key
        ]
        if len(contains) == 1:
            index, name = contains[0]
            result[name] = index
            continue

        suggestions = difflib.get_close_matches(
            requested, list(disease_names), n=8, cutoff=0.35,
        )
        raise ValueError(
            f"Target disease not uniquely found: {requested!r}. "
            f"Close matches: {suggestions}"
        )
    return result


def make_strict_training_pairs(
    association: np.ndarray,
    seed: int,
    excluded_negative_diseases: Set[int],
    heldout_positive_edges: Set[Tuple[int, int]],
    negative_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    if negative_ratio <= 0:
        raise ValueError("negative_ratio must be > 0")
    association = np.asarray(association)
    positive_m, positive_d = np.where(association > 0)
    keep = np.ones(len(positive_m), dtype=np.bool_)
    if heldout_positive_edges:
        held = set((int(m), int(d)) for m, d in heldout_positive_edges)
        for idx, (m, d) in enumerate(zip(positive_m, positive_d)):
            if (int(m), int(d)) in held:
                keep[idx] = False
    positive_m = positive_m[keep].astype(np.int64)
    positive_d = positive_d[keep].astype(np.int64)

    unknown_m, unknown_d = np.where(association == 0)
    if excluded_negative_diseases:
        excluded = np.asarray(sorted(excluded_negative_diseases), dtype=np.int64)
        eligible = ~np.isin(unknown_d, excluded)
        unknown_m = unknown_m[eligible]
        unknown_d = unknown_d[eligible]

    desired_negative = int(round(len(positive_m) * float(negative_ratio)))
    if desired_negative < 1:
        raise ValueError("No negative examples requested after applying negative_ratio.")
    if desired_negative > len(unknown_m):
        raise ValueError(
            f"Not enough eligible unknown pairs for negative sampling: need {desired_negative}, "
            f"available {len(unknown_m)} after excluding target diseases."
        )

    rng = np.random.default_rng(seed)
    selected_negative = rng.choice(len(unknown_m), size=desired_negative, replace=False)
    negative_m = unknown_m[selected_negative].astype(np.int64)
    negative_d = unknown_d[selected_negative].astype(np.int64)

    m = np.concatenate([positive_m, negative_m])
    d = np.concatenate([positive_d, negative_d])
    y = np.concatenate([
        np.ones(len(positive_m), dtype=np.int64),
        np.zeros(len(negative_m), dtype=np.int64),
    ])
    order = rng.permutation(len(y))
    m, d, y = m[order], d[order], y[order]

    # Audits that should never fail.
    for disease_index in excluded_negative_diseases:
        if np.any((d == int(disease_index)) & (y == 0)):
            raise RuntimeError(
                "Strict negative-sampling audit failed: a target-disease unknown pair "
                "entered the training set as a negative."
            )
    if heldout_positive_edges:
        label_lookup = {(int(mm), int(dd)): int(yy) for mm, dd, yy in zip(m, d, y)}
        leaked = [edge for edge in heldout_positive_edges if edge in label_lookup]
        if leaked:
            raise RuntimeError(
                f"Held-positive leakage audit failed; {len(leaked)} held edges remain in training."
            )

    stats = {
        "positive_train": int((y == 1).sum()),
        "negative_train": int((y == 0).sum()),
        "total_train": int(len(y)),
        "eligible_negative_pool": int(len(unknown_m)),
        "heldout_positive_count": int(len(heldout_positive_edges)),
        "excluded_negative_disease_count": int(len(excluded_negative_diseases)),
    }
    return m, d, y, stats

@dataclass
class FittedCaseModel:
    bio_model: Any
    bio_ema: Any
    structural_stage: Any
    structural_expert: Any
    completion_model: Any
    fusion_weights: np.ndarray
    selection: Dict[str, Any]


def fit_case_model(M, data, train_arrays, seed: int) -> FittedCaseModel:
    labels = np.asarray(train_arrays[2], dtype=np.int64)
    if len(np.unique(labels)) != 2:
        raise ValueError("Training data must contain both positive and negative labels.")

    inner_index, validation_index = M.make_inner_split(labels, seed + 1009)
    inner_arrays = tuple(np.asarray(value)[inner_index] for value in train_arrays)
    validation_arrays = tuple(np.asarray(value)[validation_index] for value in train_arrays)
    log(
        f"Model selection split: inner={len(inner_index)}, validation={len(validation_index)}, "
        f"validation positives={int(validation_arrays[2].sum())}"
    )

    bio_epoch, bio_inner_auc, _, validation_bio = M.select_bio_epoch(
        data, inner_arrays, validation_arrays, seed + 4001,
    )

    inner_stage = M.build_stage_data(data.association.shape, inner_arrays, seed + 4507)
    inner_expert = M.fit_structural_expert(inner_stage, inner_arrays, data, seed + 4513)
    validation_structure = M.predict_structural_expert(
        inner_expert, inner_stage, validation_arrays, data,
    )
    structure_inner_auc = float(M.safe_auc(validation_arrays[2], validation_structure))

    completion_epoch, completion_inner_auc, _, validation_completion = M.select_completion_epoch(
        data, inner_arrays, validation_arrays, seed + 4703,
    )

    fusion_weights, fusion_inner_auc, _ = M.select_fusion_weights(
        validation_arrays[2], validation_bio, validation_structure, validation_completion,
    )
    log(
        "Selected model: "
        f"BIO epoch={bio_epoch} (AUC={bio_inner_auc:.4f}), "
        f"STRUCT AUC={structure_inner_auc:.4f}, "
        f"COMP epoch={completion_epoch} (AUC={completion_inner_auc:.4f}), "
        f"fusion=[{fusion_weights[0]:.2f}, {fusion_weights[1]:.2f}, {fusion_weights[2]:.2f}] "
        f"(AUC={fusion_inner_auc:.4f})"
    )

    del inner_stage, inner_expert
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bio_model, bio_ema, _ = M.fit_bio_from_scratch(
        data, train_arrays, int(bio_epoch), seed + 5003,
    )
    structural_stage = M.build_stage_data(
        data.association.shape, train_arrays, seed + 6007,
    )
    structural_expert = M.fit_structural_expert(
        structural_stage, train_arrays, data, seed + 6011,
    )
    completion_model, _ = M.fit_completion_from_scratch(
        data, train_arrays, int(completion_epoch), seed + 7001,
    )

    selection = {
        "bio_epoch": int(bio_epoch),
        "bio_inner_auc": float(bio_inner_auc),
        "structure_inner_auc": float(structure_inner_auc),
        "completion_epoch": int(completion_epoch),
        "completion_inner_auc": float(completion_inner_auc),
        "fusion_weights": [float(x) for x in fusion_weights],
        "fusion_inner_auc": float(fusion_inner_auc),
        "inner_train_size": int(len(inner_index)),
        "validation_size": int(len(validation_index)),
    }
    return FittedCaseModel(
        bio_model=bio_model,
        bio_ema=bio_ema,
        structural_stage=structural_stage,
        structural_expert=structural_expert,
        completion_model=completion_model,
        fusion_weights=np.asarray(fusion_weights, dtype=np.float32),
        selection=selection,
    )


def cleanup_fitted(model: FittedCaseModel) -> None:
    del model.bio_model
    del model.bio_ema
    del model.structural_stage
    del model.structural_expert
    del model.completion_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_branch_scores(M, fitted: FittedCaseModel, data, mirna_indices, disease_indices):
    mirna_indices = np.asarray(mirna_indices, dtype=np.int64)
    disease_indices = np.asarray(disease_indices, dtype=np.int64)
    dummy = np.zeros(len(mirna_indices), dtype=np.int64)
    arrays = (mirna_indices, disease_indices, dummy)
    bio = M.predict_bio(fitted.bio_model, arrays, fitted.bio_ema)
    structure = M.predict_structural_expert(
        fitted.structural_expert, fitted.structural_stage, arrays, data,
    )
    completion = M.predict_completion(fitted.completion_model, arrays)
    return (
        np.asarray(bio, dtype=np.float32),
        np.asarray(structure, dtype=np.float32),
        np.asarray(completion, dtype=np.float32),
    )


def predict_ranked_candidate_pool(M, fitted: FittedCaseModel, data, disease_index: int, mirna_indices):
    mirna_indices = np.asarray(mirna_indices, dtype=np.int64)
    diseases = np.full(len(mirna_indices), int(disease_index), dtype=np.int64)
    bio, structure, completion = predict_branch_scores(
        M, fitted, data, mirna_indices, diseases,
    )
    fusion = M.rank_fuse_three(bio, structure, completion, fitted.fusion_weights)
    order = np.argsort(-fusion, kind="mergesort")
    return {
        "mirna_indices": mirna_indices,
        "bio": bio,
        "structure": structure,
        "completion": completion,
        "fusion": np.asarray(fusion, dtype=np.float32),
        "order": order,
    }

def prospective_rows_for_disease(
    M,
    fitted: FittedCaseModel,
    data,
    disease_name: str,
    disease_index: int,
    topk: int,
    hmdd: Mapping[Tuple[str, str], HMDDEvidence],
    aliases: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    association_column = np.asarray(data.association[:, disease_index], dtype=np.int64)
    candidate_indices = np.flatnonzero(association_column == 0).astype(np.int64)
    known_indices = np.flatnonzero(association_column > 0).astype(np.int64)
    if len(candidate_indices) == 0:
        raise ValueError(f"Disease {disease_name!r} has no benchmark-unknown candidate pairs.")

    ranked = predict_ranked_candidate_pool(
        M, fitted, data, disease_index, candidate_indices,
    )
    candidate_rows: List[Dict[str, Any]] = []
    for rank, local_idx in enumerate(ranked["order"], start=1):
        mirna_index = int(ranked["mirna_indices"][local_idx])
        mirna_name = str(data.mirna_names[mirna_index])
        row = {
            "CandidateRank": rank,
            "miRNA_Index": mirna_index,
            "miRNA": mirna_name,
            "Disease_Index": int(disease_index),
            "Disease": disease_name,
            "FusionRankScore": float(ranked["fusion"][local_idx]),
            "BiologicalBranchProbability": float(ranked["bio"][local_idx]),
            "StructuralBranchProbability": float(ranked["structure"][local_idx]),
            "CompletionBranchProbability": float(ranked["completion"][local_idx]),
            "BenchmarkKnownAssociation": "NO",
            "UsedAsNegativeInCaseStudyTraining": "NO",
            "Interpretation": "NOVEL_CANDIDATE_WRT_BENCHMARK",
        }
        row.update(hmdd_evidence_for(hmdd, mirna_name, disease_name, aliases))
        candidate_rows.append(row)

    known_rows: List[Dict[str, Any]] = []
    if len(known_indices):
        known_d = np.full(len(known_indices), int(disease_index), dtype=np.int64)
        bio, structure, completion = predict_branch_scores(
            M, fitted, data, known_indices, known_d,
        )
        for pos, mirna_index in enumerate(known_indices):
            mirna_name = str(data.mirna_names[int(mirna_index)])
            row = {
                "miRNA_Index": int(mirna_index),
                "miRNA": mirna_name,
                "Disease_Index": int(disease_index),
                "Disease": disease_name,
                "BiologicalBranchProbability": float(bio[pos]),
                "StructuralBranchProbability": float(structure[pos]),
                "CompletionBranchProbability": float(completion[pos]),
                "BenchmarkKnownAssociation": "YES",
                "TrainingStatus": "POSITIVE_EDGE_USED_IN_TRAINING",
                "Interpretation": "TRAINING_EXPOSED_NOT_A_NOVEL_CANDIDATE",
            }
            row.update(hmdd_evidence_for(hmdd, mirna_name, disease_name, aliases))
            known_rows.append(row)

    top = candidate_rows[: min(topk, len(candidate_rows))]
    top_hmdd = int(sum(row.get("HMDD_Corroborated") == "YES" for row in top))
    all_hmdd = int(sum(row.get("HMDD_Corroborated") == "YES" for row in candidate_rows))
    metadata = {
        "disease": disease_name,
        "disease_index": int(disease_index),
        "known_association_count": int(len(known_indices)),
        "novel_candidate_count": int(len(candidate_indices)),
        "topk_requested": int(topk),
        "topk_returned": int(len(top)),
        "hmdd_corroborated_in_topk": top_hmdd,
        "hmdd_corroborated_in_all_candidates": all_hmdd,
        "note": (
            "Known benchmark positives were excluded from the candidate ranking. "
            "All benchmark-unknown pairs for target diseases were excluded from negative sampling. "
            "FusionRankScore is a rank-fusion score, not a calibrated probability."
        ),
    }
    if hmdd and len(top) and len(candidate_rows):
        top_rate = top_hmdd / len(top)
        all_rate = all_hmdd / len(candidate_rows)
        metadata.update({
            "HMDD_CorroborationRate_TopK": float(top_rate),
            "HMDD_CorroborationRate_AllCandidates": float(all_rate),
            "HMDD_EnrichmentFold_TopK_vs_All": float(top_rate / all_rate) if all_rate > 0 else None,
            "HMDD_HypergeometricP_TopK_Enrichment": hypergeometric_survival(
                top_hmdd, len(candidate_rows), all_hmdd, len(top),
            ),
            "HMDD_EnrichmentInterpretation": (
                "Post-hoc database corroboration enrichment only; not a temporally independent prospective validation."
            ),
        })
    return candidate_rows, known_rows, metadata

def choose_heldout_edges(
    association: np.ndarray,
    disease_index: int,
    holdout_fraction: float,
    seed: int,
    min_retained: int,
) -> Tuple[Set[Tuple[int, int]], np.ndarray]:
    known = np.flatnonzero(np.asarray(association[:, disease_index]) > 0).astype(np.int64)
    if len(known) == 0:
        return set(), known
    if not 0.0 < holdout_fraction <= 1.0:
        raise ValueError("retrospective_holdout_fraction must be in (0, 1].")
    max_holdout = len(known) - max(0, min(int(min_retained), len(known)))
    if max_holdout <= 0:
        # For a disease with only one known edge, holding it out is more informative than skipping.
        max_holdout = len(known)
    count = max(1, int(round(len(known) * holdout_fraction)))
    count = min(count, max_holdout)
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(known, size=count, replace=False).astype(np.int64))
    held = {(int(m), int(disease_index)) for m in selected}
    return held, selected


def ndcg_at_k(ranks: Sequence[int], relevant_count: int, k: int) -> float:
    if relevant_count <= 0:
        return float("nan")
    dcg = sum(1.0 / math.log2(rank + 1.0) for rank in ranks if rank <= k)
    ideal_count = min(relevant_count, k)
    idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_count + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_heldout_ranks(ranks: Sequence[int], candidate_count: int) -> Dict[str, Any]:
    ranks = [int(rank) for rank in ranks]
    if not ranks:
        return {}
    metrics: Dict[str, Any] = {
        "HeldPositiveCount": len(ranks),
        "EligibleCandidateCount": int(candidate_count),
        "BestHeldPositiveRank": int(min(ranks)),
        "MedianHeldPositiveRank": float(np.median(ranks)),
        "MeanHeldPositiveRank": float(np.mean(ranks)),
        "MeanReciprocalRankOfHeldEdges": float(np.mean([1.0 / rank for rank in ranks])),
        "MeanHeldPositivePercentile": float(np.mean([
            1.0 - (rank - 1) / max(candidate_count - 1, 1) for rank in ranks
        ])),
    }
    for k in (10, 20, 50, 100):
        recovered = sum(rank <= k for rank in ranks)
        recall = float(recovered / len(ranks))
        random_expected_recall = float(min(k, candidate_count) / max(candidate_count, 1))
        metrics[f"RecoveredAt{k}"] = int(recovered)
        metrics[f"RecallAt{k}"] = recall
        metrics[f"HitAt{k}"] = int(recovered > 0)
        metrics[f"NDCGAt{k}"] = ndcg_at_k(ranks, len(ranks), k)
        metrics[f"RandomExpectedRecallAt{k}"] = random_expected_recall
        metrics[f"RecoveryLiftAt{k}"] = (
            float(recall / random_expected_recall) if random_expected_recall > 0 else float("nan")
        )
    return metrics


def run_retrospective_for_target(
    M,
    data,
    disease_name: str,
    disease_index: int,
    args: argparse.Namespace,
    hmdd: Mapping[Tuple[str, str], HMDDEvidence],
    aliases: Mapping[str, str],
    disease_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    repeat_summaries: List[Dict[str, Any]] = []
    combined_held_rows: List[Dict[str, Any]] = []
    known_count = int(np.asarray(data.association[:, disease_index] > 0).sum())
    if known_count == 0:
        log(f"Retrospective skipped for {disease_name}: no known benchmark positives.")
        return repeat_summaries, combined_held_rows

    for repeat in range(1, args.retrospective_repeats + 1):
        repeat_seed = int(args.seed + disease_index * 100_003 + repeat * 10_007)
        log(
            f"Retrospective {disease_name}: repeat {repeat}/{args.retrospective_repeats}, "
            f"seed={repeat_seed}"
        )
        held_edges, held_mirnas = choose_heldout_edges(
            data.association,
            disease_index,
            args.retrospective_holdout_fraction,
            repeat_seed,
            args.retrospective_min_retained_known,
        )
        if not held_edges:
            log(f"Retrospective skipped for {disease_name}: no edge could be held out.")
            break

        train_m, train_d, train_y, sampling_stats = make_strict_training_pairs(
            data.association,
            seed=repeat_seed + 17,
            excluded_negative_diseases={int(disease_index)},
            heldout_positive_edges=held_edges,
            negative_ratio=args.negative_ratio,
        )
        fitted = fit_case_model(M, data, (train_m, train_d, train_y), repeat_seed + 29)

        association_column = np.asarray(data.association[:, disease_index], dtype=np.int64)
        retained_known = set(np.flatnonzero(association_column > 0).astype(int)) - set(map(int, held_mirnas))
        eligible = np.asarray(
            [index for index in range(len(data.mirna_names)) if index not in retained_known],
            dtype=np.int64,
        )
        ranked = predict_ranked_candidate_pool(M, fitted, data, disease_index, eligible)
        rank_by_mirna = {
            int(ranked["mirna_indices"][local_idx]): rank
            for rank, local_idx in enumerate(ranked["order"], start=1)
        }
        local_by_mirna = {
            int(m): pos for pos, m in enumerate(ranked["mirna_indices"])
        }

        held_rows: List[Dict[str, Any]] = []
        ranks: List[int] = []
        for mirna_index in held_mirnas:
            mirna_index = int(mirna_index)
            rank = int(rank_by_mirna[mirna_index])
            ranks.append(rank)
            pos = int(local_by_mirna[mirna_index])
            mirna_name = str(data.mirna_names[mirna_index])
            row = {
                "Repeat": int(repeat),
                "HeldPositiveRank": rank,
                "miRNA_Index": mirna_index,
                "miRNA": mirna_name,
                "Disease_Index": int(disease_index),
                "Disease": disease_name,
                "FusionRankScore": float(ranked["fusion"][pos]),
                "BiologicalBranchProbability": float(ranked["bio"][pos]),
                "StructuralBranchProbability": float(ranked["structure"][pos]),
                "CompletionBranchProbability": float(ranked["completion"][pos]),
                "BenchmarkKnownAssociation": "YES",
                "TrainingStatus": "HELD_OUT_BEFORE_MODEL_SELECTION_AND_REFIT",
                "UsedAsNegativeInCaseStudyTraining": "NO",
            }
            row.update(hmdd_evidence_for(hmdd, mirna_name, disease_name, aliases))
            held_rows.append(row)
            combined_held_rows.append(row)

        metrics = evaluate_heldout_ranks(ranks, len(eligible))
        summary = {
            "Disease": disease_name,
            "DiseaseIndex": int(disease_index),
            "Repeat": int(repeat),
            "Seed": int(repeat_seed),
            "OriginalKnownAssociationCount": known_count,
            "HeldPositiveCount": int(len(held_edges)),
            "RetainedKnownPositiveCount": int(known_count - len(held_edges)),
            **sampling_stats,
            **fitted.selection,
            **metrics,
            "Protocol": (
                "Held positives removed before inner model selection and final refit; "
                "all benchmark-unknown pairs for this target disease excluded from negative sampling; "
                "retained known positives excluded from the recovery ranking pool."
            ),
        }
        repeat_summaries.append(summary)
        write_csv(disease_dir / f"retrospective_repeat_{repeat}_held_edges.csv", held_rows)
        write_json(disease_dir / f"retrospective_repeat_{repeat}_summary.json", summary)
        cleanup_fitted(fitted)

    if repeat_summaries:
        # Aggregate only numeric recovery metrics that have stable meaning across repeats.
        aggregate: Dict[str, Any] = {
            "Disease": disease_name,
            "Repeats": len(repeat_summaries),
        }
        keys = [
            "BestHeldPositiveRank", "MedianHeldPositiveRank", "MeanHeldPositiveRank",
            "MeanReciprocalRankOfHeldEdges", "MeanHeldPositivePercentile",
            "RecallAt10", "RecallAt20", "RecallAt50", "RecallAt100",
            "HitAt10", "HitAt20", "HitAt50", "HitAt100",
            "NDCGAt10", "NDCGAt20", "NDCGAt50", "NDCGAt100",
        ]
        for key in keys:
            values = [float(row[key]) for row in repeat_summaries if key in row]
            if values:
                aggregate[f"{key}_mean"] = float(np.mean(values))
                aggregate[f"{key}_std"] = float(np.std(values))
        write_json(disease_dir / "retrospective_aggregate_summary.json", aggregate)

    return repeat_summaries, combined_held_rows

def save_fitted_case_model(M, fitted: FittedCaseModel, output_dir: Path, tag: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "raw_trainable_state": {
            name: value.detach().cpu() for name, value in fitted.bio_model.named_parameters()
        },
        "ema_state": fitted.bio_ema.state_dict(),
        "selection": fitted.selection,
        "config": asdict(M.CONFIG),
    }, output_dir / f"{tag}_biological_branch.pt")
    with (output_dir / f"{tag}_structural_expert.pkl").open("wb") as handle:
        pickle.dump(fitted.structural_expert, handle)
    np.savez_compressed(
        output_dir / f"{tag}_structural_stage.npz",
        association=fitted.structural_stage.association,
        mirna_gip=fitted.structural_stage.mirna_gip,
        disease_gip=fitted.structural_stage.disease_gip,
    )
    torch.save({
        "completion_state_dict": {
            name: value.detach().cpu() for name, value in fitted.completion_model.state_dict().items()
        },
        "selection": fitted.selection,
        "feature_names": list(fitted.completion_model.feature_names),
        "full_view_id": int(fitted.completion_model.full_view_id),
        "config": asdict(M.CONFIG),
    }, output_dir / f"{tag}_completion_branch.pt")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rigorous disease-centric case study for the strict three-branch MDA model."
    )
    parser.add_argument("--model", required=True, help="Path to the original strict three-branch .py model file")
    parser.add_argument("--targets", nargs="+", required=True, help="Disease names exactly/as nearly as stored by the model")
    parser.add_argument("--topk", type=int, default=20, help="Top-K novel candidates highlighted per disease")
    parser.add_argument("--out_dir", default="./StrictThreeBranch_CaseStudy", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Case-study seed; default uses model CONFIG.seed")
    parser.add_argument(
        "--negative_ratio", type=float, default=1.0,
        help="Unknown-negative / positive ratio for training. Default 1.0 matches the original model's 1:1 benchmark."
    )

    parser.add_argument("--hmdd_xlsx", default=None, help="Optional HMDD xlsx with miRNA,disease[,PMID] columns")
    parser.add_argument("--dbdemc_txt", default=None, help="Optional dbDEMC dataset-list txt with CancerType column")
    parser.add_argument(
        "--disease_alias", action="append", default=[],
        help="Optional database alias, e.g. --disease_alias 'Colonic Neoplasms=colon cancer'. Repeatable."
    )

    parser.add_argument("--skip_retrospective", action="store_true", help="Run prospective ranking only")
    parser.add_argument("--retrospective_holdout_fraction", type=float, default=0.20)
    parser.add_argument("--retrospective_repeats", type=int, default=1)
    parser.add_argument(
        "--retrospective_min_retained_known", type=int, default=1,
        help="Try to retain at least this many known target-disease positives in each retrospective training graph."
    )
    parser.add_argument("--save_models", action="store_true", help="Save the prospective fitted branch states/expert")

    # These environment settings are consumed by the original model at import time.
    parser.add_argument("--data_dir", default=None, help="Sets MDA_DATA_DIR before importing the model")
    parser.add_argument("--d14_cache_dir", default=None, help="Sets MDA_D14_CACHE_DIR before importing the model")
    parser.add_argument("--foundation_cache_dir", default=None, help="Sets MDA_FOUNDATION_CACHE_DIR before importing the model")
    parser.add_argument("--external_cache_dir", default=None, help="Sets MDA_EXTERNAL_CACHE_DIR before importing the model")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.topk <= 0:
        raise ValueError("--topk must be positive")
    if args.retrospective_repeats <= 0:
        raise ValueError("--retrospective_repeats must be positive")

    configure_model_environment(args)
    model_path = Path(args.model)
    M = load_model_module(model_path)
    if args.seed is None:
        args.seed = int(M.CONFIG.seed)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    aliases = parse_aliases(args.disease_alias)

    log("=" * 88)
    log("Strict three-branch disease-centric case study started")
    log(f"Model: {model_path.resolve()}")
    log(f"Device: {M.DEVICE}")
    log(f"Case-study seed: {args.seed}")
    log("Loading model data through the original model's audited loader")
    M.set_seed(int(args.seed))
    data = M.load_model_data()
    log(
        f"Loaded miRNAs={len(data.mirna_names)}, diseases={len(data.disease_names)}, "
        f"known benchmark associations={int(np.asarray(data.association).sum())}"
    )

    resolved_targets = resolve_target_indices(data.disease_names, args.targets)
    log("Resolved targets: " + ", ".join(f"{name} [index={idx}]" for name, idx in resolved_targets.items()))

    hmdd = load_hmdd_pairs(Path(args.hmdd_xlsx).expanduser().resolve() if args.hmdd_xlsx else None)
    dbdemc_types = load_dbdemc_cancer_types(
        Path(args.dbdemc_txt).expanduser().resolve() if args.dbdemc_txt else None
    )
    if hmdd:
        log(f"Loaded HMDD normalized pair keys: {len(hmdd)}")
    if dbdemc_types:
        log(f"Loaded dbDEMC disease/cancer-type entries: {len(dbdemc_types)}")

    target_indices = set(int(index) for index in resolved_targets.values())
    log("Building strict prospective training pairs")
    train_m, train_d, train_y, prospective_sampling = make_strict_training_pairs(
        data.association,
        seed=int(args.seed) + 101,
        excluded_negative_diseases=target_indices,
        heldout_positive_edges=set(),
        negative_ratio=float(args.negative_ratio),
    )
    log(
        f"Prospective training pairs: {prospective_sampling['positive_train']} positives + "
        f"{prospective_sampling['negative_train']} negatives; target-disease unknowns excluded"
    )
    prospective_model = fit_case_model(
        M, data, (train_m, train_d, train_y), int(args.seed) + 211,
    )

    if args.save_models:
        save_fitted_case_model(M, prospective_model, out_dir / "model_states", "prospective")

    excel_sheets: Dict[str, Sequence[Mapping[str, Any]]] = {}
    prospective_summary_rows: List[Dict[str, Any]] = []
    retrospective_summary_rows: List[Dict[str, Any]] = []
    retrospective_held_rows: List[Dict[str, Any]] = []

    for disease_name, disease_index in resolved_targets.items():
        disease_dir = out_dir / safe_name(disease_name)
        disease_dir.mkdir(parents=True, exist_ok=True)
        log(f"Prospective ranking: {disease_name}")
        candidates, known, metadata = prospective_rows_for_disease(
            M, prospective_model, data,
            disease_name=disease_name,
            disease_index=int(disease_index),
            topk=int(args.topk),
            hmdd=hmdd,
            aliases=aliases,
        )
        coverage = dbdemc_disease_coverage(disease_name, dbdemc_types, aliases)
        metadata["dbDEMC_DiseaseLevelCoverage"] = (
            "YES" if dbdemc_types and coverage else ("NO" if dbdemc_types else "")
        )
        metadata["dbDEMC_Interpretation"] = (
            "Disease-level dataset coverage only; deliberately NOT assigned as miRNA-pair evidence."
            if dbdemc_types else ""
        )
        metadata.update({
            "prospective_training_sampling": prospective_sampling,
            "model_selection": prospective_model.selection,
        })

        write_csv(disease_dir / "prospective_all_novel_candidates.csv", candidates)
        write_csv(disease_dir / f"prospective_top{args.topk}_novel_candidates.csv", candidates[: args.topk])
        write_csv(disease_dir / "known_training_associations_separate.csv", known)
        write_json(disease_dir / "prospective_metadata.json", metadata)

        prospective_summary_rows.append({
            "Disease": disease_name,
            "DiseaseIndex": int(disease_index),
            "KnownTrainingAssociations": metadata["known_association_count"],
            "NovelCandidateCount": metadata["novel_candidate_count"],
            f"HMDD_Corroborated_in_Top{args.topk}": metadata["hmdd_corroborated_in_topk"],
            "HMDD_Corroborated_AllCandidates": metadata["hmdd_corroborated_in_all_candidates"],
            "HMDD_EnrichmentFold_TopK_vs_All": metadata.get("HMDD_EnrichmentFold_TopK_vs_All", ""),
            "HMDD_HypergeometricP_TopK_Enrichment": metadata.get("HMDD_HypergeometricP_TopK_Enrichment", ""),
            "dbDEMC_DiseaseLevelCoverage": metadata["dbDEMC_DiseaseLevelCoverage"],
            "FusionWeight_Bio": prospective_model.selection["fusion_weights"][0],
            "FusionWeight_Structure": prospective_model.selection["fusion_weights"][1],
            "FusionWeight_Completion": prospective_model.selection["fusion_weights"][2],
            "FusionInnerValidationAUC": prospective_model.selection["fusion_inner_auc"],
        })
        excel_sheets[f"{safe_name(disease_name, 20)}_Top{args.topk}"] = candidates[: args.topk]
        excel_sheets[f"{safe_name(disease_name, 20)}_Known"] = known

    cleanup_fitted(prospective_model)

    if not args.skip_retrospective:
        for disease_name, disease_index in resolved_targets.items():
            disease_dir = out_dir / safe_name(disease_name)
            summaries, held_rows = run_retrospective_for_target(
                M, data,
                disease_name=disease_name,
                disease_index=int(disease_index),
                args=args,
                hmdd=hmdd,
                aliases=aliases,
                disease_dir=disease_dir,
            )
            retrospective_summary_rows.extend(summaries)
            retrospective_held_rows.extend(held_rows)
            if held_rows:
                excel_sheets[f"{safe_name(disease_name, 18)}_Held"] = held_rows

    write_csv(out_dir / "prospective_summary.csv", prospective_summary_rows)
    if retrospective_summary_rows:
        write_csv(out_dir / "retrospective_repeat_summaries.csv", retrospective_summary_rows)
    if retrospective_held_rows:
        write_csv(out_dir / "retrospective_all_held_edges.csv", retrospective_held_rows)

    study_manifest = {
        "model_file": str(model_path.resolve()),
        "output_dir": str(out_dir),
        "seed": int(args.seed),
        "targets": resolved_targets,
        "topk": int(args.topk),
        "negative_ratio": float(args.negative_ratio),
        "retrospective_enabled": not bool(args.skip_retrospective),
        "retrospective_holdout_fraction": float(args.retrospective_holdout_fraction),
        "retrospective_repeats": int(args.retrospective_repeats),
        "retrospective_min_retained_known": int(args.retrospective_min_retained_known),
        "model_config": asdict(M.CONFIG),
        "prospective_sampling": prospective_sampling,
        "methodological_safeguards": [
            "Known benchmark positives are exported separately and excluded from novel-candidate ranking.",
            "All unknown pairs for requested target diseases are excluded from negative sampling.",
            "Retrospective held positives are removed before inner model selection and final branch refitting.",
            "Structural/completion cross-fitted graph views are reused from the original model implementation.",
            "HMDD is post-hoc pair-level corroboration and does not affect training/ranking.",
            "dbDEMC dataset-list coverage is reported only at disease level, not as pair-level evidence.",
            "FusionRankScore is explicitly treated as a rank score rather than a calibrated probability.",
        ],
    }
    write_json(out_dir / "case_study_manifest.json", study_manifest)
    excel_sheets["Prospective_Summary"] = prospective_summary_rows
    if retrospective_summary_rows:
        excel_sheets["Retrospective_Summary"] = retrospective_summary_rows
    maybe_write_excel(out_dir / "CaseStudy_Combined.xlsx", excel_sheets)

    log(f"Case study completed. Outputs: {out_dir}")
    log("=" * 88)


if __name__ == "__main__":
    main()
