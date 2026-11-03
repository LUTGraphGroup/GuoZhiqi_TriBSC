from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "main_01.py"
CASE_PIPELINE_PATH = SCRIPT_DIR / "strict_three_branch_case_study.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "Main01_Final_CaseStudy_Output"
DEFAULT_TARGETS = ("Colonic Neoplasms", "Liver Neoplasms")
DEFAULT_TOPK = 15


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_case_pipeline(path: Path):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Case-study pipeline not found: {path}\n"
            "Put strict_three_branch_case_study.py beside this adapter."
        )
    module_name = "main01_case_study_pipeline"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load case-study pipeline from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "main") or not hasattr(module, "load_model_module"):
        raise AttributeError(
            "strict_three_branch_case_study.py does not expose the expected "
            "main/load_model_module interface."
        )
    return module


def verify_final_model_contract(model) -> None:
    expected_b = (
        "path3_normalized",
        "path3_raw",
        "path3_resource_allocation",
        "static_left",
        "static_right",
        "gip_left",
        "gip_both",
        "mirna_degree",
        "disease_degree",
    )
    expected_c_propagation = (
        "static_left",
        "static_right",
        "functional_left",
        "functional_semantic_both",
    )
    expected_c_auxiliary = (
        "mirna_degree",
        "disease_degree",
        "degree_product",
    )
    checks = {
        "STRUCTURAL_FEATURE_NAMES": expected_b,
        "COMPLETION_PROPAGATION_NAMES": expected_c_propagation,
        "COMPLETION_AUXILIARY_NAMES": expected_c_auxiliary,
    }
    for attribute, expected in checks.items():
        actual = getattr(model, attribute, None)
        if actual is None or tuple(actual) != expected:
            raise RuntimeError(
                f"The selected model is not the final main_01 version: "
                f"{attribute} expected {expected}, got {actual}."
            )

    ranks = tuple(int(value) for value in model.CONFIG.completion_score_ranks)
    if ranks != (8, 16, 32, 64):
        raise RuntimeError(
            "The selected model has unexpected completion ranks; expected "
            f"(8, 16, 32, 64), got {ranks}."
        )
    if not hasattr(model, "select_fusion_weights") or not hasattr(
        model, "rank_fuse_three"
    ):
        raise RuntimeError("The final validation-guided rank-fusion API is missing.")


def has_option(arguments: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(option + "=") for value in arguments)


def option_value(arguments: Sequence[str], option: str, default: str) -> str:
    for index, value in enumerate(arguments):
        if value == option and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(option + "="):
            return value.split("=", 1)[1]
    return default


def apply_defaults(arguments: Sequence[str]) -> List[str]:
    result = list(arguments)
    if not has_option(result, "--model"):
        result.extend(["--model", str(MODEL_PATH)])
    if not has_option(result, "--targets"):
        # Append nargs='+' last so it cannot consume later optional arguments.
        result.extend(["--targets", *DEFAULT_TARGETS])
    if not has_option(result, "--topk"):
        result.extend(["--topk", str(DEFAULT_TOPK)])
    if not has_option(result, "--out_dir"):
        result.extend(["--out_dir", str(DEFAULT_OUTPUT_DIR)])
    return result


def main() -> None:
    if not MODEL_PATH.is_file() and not has_option(sys.argv[1:], "--model"):
        raise FileNotFoundError(
            f"Final model not found: {MODEL_PATH}\n"
            "Put main_01.py beside this adapter or provide --model PATH."
        )

    pipeline = load_case_pipeline(CASE_PIPELINE_PATH)
    original_loader = pipeline.load_model_module

    def load_verified_model(path: Path):
        model = original_loader(path)
        verify_final_model_contract(model)
        pipeline.log(
            "Final-model contract verified: branch B=9 features; "
            "branch C=8 SVD + 4 propagation + 3 auxiliary features; "
            "validation-guided rank fusion enabled."
        )
        return model

    pipeline.load_model_module = load_verified_model
    forwarded = apply_defaults(sys.argv[1:])
    sys.argv = [str(Path(__file__).resolve()), *forwarded]
    pipeline.main()

    model_value = option_value(forwarded, "--model", str(MODEL_PATH))
    output_value = option_value(forwarded, "--out_dir", str(DEFAULT_OUTPUT_DIR))
    model_path = Path(model_value).expanduser().resolve()
    output_dir = Path(output_value).expanduser().resolve()
    adapter_manifest = {
        "adapter": str(Path(__file__).resolve()),
        "adapter_sha256": file_sha256(Path(__file__).resolve()),
        "case_pipeline": str(CASE_PIPELINE_PATH.resolve()),
        "case_pipeline_sha256": file_sha256(CASE_PIPELINE_PATH.resolve()),
        "model": str(model_path),
        "model_sha256": file_sha256(model_path),
        "model_contract": {
            "branch_B_feature_count": 9,
            "branch_C_SVD_channel_count": 8,
            "branch_C_propagation_channel_count": 4,
            "branch_C_auxiliary_feature_count": 3,
            "fusion": (
                "rank normalization followed by complete non-negative simplex "
                "search on inner validation only"
            ),
        },
        "effective_command_arguments": forwarded,
        "default_targets_if_not_overridden": list(DEFAULT_TARGETS),
        "default_topk_if_not_overridden": DEFAULT_TOPK,
        "interpretation": (
            "FusionRankScore is a within-candidate-pool rank-fusion score, "
            "not a calibrated biological probability."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "main01_adapter_manifest.json").write_text(
        json.dumps(adapter_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

