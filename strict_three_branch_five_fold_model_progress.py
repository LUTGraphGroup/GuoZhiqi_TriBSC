from __future__ import annotations

import csv
import json
import math
import os
import pickle
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import randomized_svd
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
SERVER_ROOT = Path("/root/autodl-tmp") if os.name != "nt" else SCRIPT_DIR
D14_DIR = Path(os.environ.get("MDA_D14_CACHE_DIR", SERVER_ROOT / "D14_UnifiedDataCache"))
FOUNDATION_DIR = Path(os.environ.get("MDA_FOUNDATION_CACHE_DIR", SERVER_ROOT / "Bio_Foundation_Cache"))
EXTERNAL_CACHE_DIR = Path(os.environ.get(
    "MDA_EXTERNAL_CACHE_DIR", SERVER_ROOT / "D16_BioHeteroPath_Cache",
))

PREPARE_ONLY: bool = False


DISEASE_FEATURE_SIMILARITY_FILENAME: str = "disease_feature_vector_similarity_matrix.txt"
OUTPUT_DIRECTORY_NAME: str = "StrictThreeBranchFiveFold_Output"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 10
    folds: int = 5
    inner_validation_ratio: float = 0.20
    crossfit_graph_views: int = 5

    branch_dim: int = 128
    gene_hidden_dim: int = 96
    endpoint_hidden_dim: int = 128
    gene_cross_hidden_dim: int = 128
    dropout: float = 0.16
    gene_modality_dropout: float = 0.08
    endpoint_gene_gate_bias: float = -1.0

    batch_size: int = 128
    eval_batch_size: int = 512
    maximum_epochs: int = 55
    patience: int = 8
    bio_learning_rate: float = 8e-5
    weight_decay: float = 2e-4
    gradient_clip: float = 2.0
    use_amp: bool = True


    bio_ranking_weight: float = 0.30
    bio_ranking_temperature: float = 0.25
    bio_ema_decay: float = 0.995
    bio_early_stop_min_delta: float = 1e-4

    similarity_topk: int = 15
    expert_max_iter: int = 200
    expert_learning_rate: float = 0.04
    expert_max_leaf_nodes: int = 15
    expert_min_samples_leaf: int = 30
    expert_l2_regularization: float = 3.0


    completion_score_ranks: Tuple[int, ...] = (8,16,32,64)
    completion_hidden_dim: int = 48
    completion_dropout: float = 0.25
    completion_batch_size: int = 256
    completion_eval_batch_size: int = 1024
    completion_maximum_epochs: int = 80
    completion_patience: int = 10
    completion_early_stop_min_delta: float = 2e-5
    completion_learning_rate: float = 4e-4
    completion_weight_decay: float = 2e-3
    completion_gradient_clip: float = 2.0
    completion_ranking_weight: float = 0.25
    completion_ranking_temperature: float = 0.25
    completion_residual_scale: float = 0.2


    fusion_grid_step: float = 0.05
    fusion_min_bio_weight: float = 0.00
    fusion_max_bio_weight: float = 0.30
    fusion_min_structure_weight: float = 0.45
    fusion_min_completion_weight: float = 0.00
    fusion_max_completion_weight: float = 0.40


CONFIG = ExperimentConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_raw_file(filename: str) -> Path:
    candidates = [
        SERVER_ROOT / filename,
        SERVER_ROOT / "data" / filename,
        SCRIPT_DIR / filename,
        SCRIPT_DIR / "data" / filename,
    ]
    environment = os.environ.get("MDA_DATA_DIR", "").strip()
    if environment:
        candidates = [Path(environment) / filename, Path(environment) / "data" / filename] + candidates
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Cannot find {filename}. Searched:\n  - " + "\n  - ".join(map(str, candidates))
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clean_similarity(array: np.ndarray, size: int, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.shape != (size, size):
        raise ValueError(f"{name} expected {(size, size)}, got {value.shape}")
    value = np.clip(np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    asymmetry = float(np.max(np.abs(value - value.T)))
    if asymmetry > 1e-4:
        raise ValueError(f"{name} is not symmetric; max asymmetry={asymmetry:.6g}")
    value = 0.5 * (value + value.T)
    np.fill_diagonal(value, 1.0)
    return value.astype(np.float32)


def load_fixed_similarity_matrix(filename: str, size: int, name: str) -> np.ndarray:
    path = resolve_raw_file(filename)
    value = np.asarray(np.loadtxt(path), dtype=np.float32)
    if value.shape != (size, size):
        raise ValueError(
            f"{name} expected {(size, size)}, got {value.shape} from {path}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity: {path}")
    minimum, maximum = float(value.min()), float(value.max())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(
            f"{name} must lie in [0, 1], got range [{minimum:.6g}, {maximum:.6g}]"
        )
    asymmetry = float(np.max(np.abs(value - value.T)))
    if asymmetry > 1e-4:
        raise ValueError(f"{name} is not symmetric; max asymmetry={asymmetry:.6g}")
    diagonal_error = float(np.max(np.abs(np.diag(value) - 1.0)))
    if diagonal_error > 1e-3:
        raise ValueError(
            f"{name} diagonal must be one; max error={diagonal_error:.6g}"
        )
    return clean_similarity(value, size, name)


def safe_auc(labels: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(labels, probability)) if len(np.unique(labels)) == 2 else 0.5


def classification_metrics(labels: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    prediction = (np.asarray(probability) >= 0.5).astype(np.int64)
    return {
        "AUC": safe_auc(labels, probability),
        "AUPR": float(average_precision_score(labels, probability)),
        "ACCURACY": float(accuracy_score(labels, prediction)),
        "PRECISION": float(precision_score(labels, prediction, zero_division=0)),
        "RECALL": float(recall_score(labels, prediction, zero_division=0)),
        "F1": float(f1_score(labels, prediction, zero_division=0)),
    }


@dataclass
class AssociationDataBundle:
    association: np.ndarray
    rna_embeddings: np.ndarray
    mirna_group_indices: np.ndarray
    mirna_group_mask: np.ndarray
    disease_embeddings: np.ndarray
    gene_context: np.ndarray
    mirna_gene_indices: np.ndarray
    mirna_gene_mask: np.ndarray
    mirna_gene_prior: np.ndarray
    disease_gene_indices: np.ndarray
    disease_gene_mask: np.ndarray
    disease_gene_prior: np.ndarray
    mirna_similarity: np.ndarray
    disease_similarity: np.ndarray
    disease_feature_similarity: np.ndarray
    overlap_features: np.ndarray
    mirna_names: List[str]
    disease_names: List[str]


def required_cache_files() -> Dict[str, Path]:
    files = {
        "manifest": D14_DIR / "manifest.json",
        "benchmark": D14_DIR / "benchmark_legacy.npz",
        "entity_names": D14_DIR / "entity_names.json",
        "rna": FOUNDATION_DIR / "rna_unique_embeddings.npy",
        "group_indices": FOUNDATION_DIR / "mirna_group_indices.npy",
        "group_mask": FOUNDATION_DIR / "mirna_group_mask.npy",
        "disease_embeddings": FOUNDATION_DIR / "disease_embeddings.npy",
        "mirna_names": FOUNDATION_DIR / "mirna_names.json",
        "disease_names": FOUNDATION_DIR / "disease_names.json",
        "external": EXTERNAL_CACHE_DIR / "new_model_external_context.npz",
        "overlap": FOUNDATION_DIR / "d6_gene_bridge_cache.npz",
    }
    missing = [path for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Standalone D21 needs the already-built data caches below; no old model script is needed:\n  - "
            + "\n  - ".join(map(str, missing))
        )
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    if manifest.get("label_safe") is not True:
        raise ValueError("D14 cache is not marked label_safe=True.")
    return files


def load_model_data() -> AssociationDataBundle:
    files = required_cache_files()
    with np.load(files["benchmark"], allow_pickle=False) as saved:
        association = saved["association"].astype(np.int64)
    rna = np.load(files["rna"]).astype(np.float32)
    group_indices = np.load(files["group_indices"]).astype(np.int64)
    group_mask = np.load(files["group_mask"]).astype(np.bool_)
    disease_embeddings = np.load(files["disease_embeddings"]).astype(np.float32)
    mirna_names = json.loads(files["mirna_names"].read_text(encoding="utf-8"))
    disease_names = json.loads(files["disease_names"].read_text(encoding="utf-8"))
    canonical_names = json.loads(files["entity_names"].read_text(encoding="utf-8"))
    if mirna_names != canonical_names["legacy_mirna_names"]:
        raise ValueError("Foundation miRNA order differs from D14 benchmark order.")
    if disease_names != canonical_names["legacy_disease_names"]:
        raise ValueError("Foundation disease order differs from D14 benchmark order.")

    with np.load(files["external"], allow_pickle=False) as saved:
        gene_context = saved["gene_context"].astype(np.float32)
        mirna_gene_indices = saved["mirna_gene_indices"].astype(np.int64)
        mirna_gene_mask = saved["mirna_gene_mask"].astype(np.bool_)
        mirna_gene_prior = saved["mirna_gene_prior"].astype(np.float32)
        disease_gene_indices = saved["disease_gene_indices"].astype(np.int64)
        disease_gene_mask = saved["disease_gene_mask"].astype(np.bool_)
        disease_gene_prior = saved["disease_gene_prior"].astype(np.float32)
        external_statistics = json.loads(str(saved["statistics"].item()))
    if external_statistics.get("uses_mirna_disease_labels") is not False:
        raise ValueError("External-context cache is not explicitly label independent.")

    with np.load(files["overlap"], allow_pickle=False) as saved:
        overlap = saved["overlap_features"].astype(np.float32)
    expected = (len(mirna_names), len(disease_names))
    if association.shape != expected:
        raise ValueError(f"Benchmark shape {association.shape} differs from entity order {expected}.")
    if disease_embeddings.shape[0] != expected[1]:
        raise ValueError("Disease embedding count is not aligned.")
    if mirna_gene_indices.shape[0] != expected[0] or disease_gene_indices.shape[0] != expected[1]:
        raise ValueError("External gene-context endpoints are not aligned.")
    if overlap.shape != (*expected, 3):
        raise ValueError(f"Gene-overlap cache expected {(*expected, 3)}, got {overlap.shape}.")

    mirna_similarity = clean_similarity(
        np.loadtxt(resolve_raw_file("miRNA_sequence_similarity.txt")),
        expected[0], "miRNA sequence similarity",
    )
    disease_similarity = clean_similarity(
        np.loadtxt(resolve_raw_file("disease_semantic_similarity.txt")),
        expected[1], "disease semantic similarity",
    )
    disease_feature_similarity = load_fixed_similarity_matrix(
        DISEASE_FEATURE_SIMILARITY_FILENAME,
        expected[1], "disease feature-vector similarity",
    )
    return AssociationDataBundle(
        association, rna, group_indices, group_mask, disease_embeddings,
        gene_context, mirna_gene_indices, mirna_gene_mask, mirna_gene_prior,
        disease_gene_indices, disease_gene_mask, disease_gene_prior,
        mirna_similarity, disease_similarity,
        disease_feature_similarity, overlap, mirna_names, disease_names,
    )


class GeneAnchoredBiologicalBranch(nn.Module):
    def __init__(self, data: AssociationDataBundle):
        super().__init__()
        self.register_buffer("rna", torch.tensor(data.rna_embeddings, dtype=torch.float32))
        self.register_buffer("groups", torch.tensor(data.mirna_group_indices, dtype=torch.long))
        self.register_buffer("group_mask", torch.tensor(data.mirna_group_mask, dtype=torch.bool))
        self.register_buffer("disease", torch.tensor(data.disease_embeddings, dtype=torch.float32))
        self.register_buffer("gene_context", torch.tensor(data.gene_context, dtype=torch.float32))
        self.register_buffer("mi", torch.tensor(data.mirna_gene_indices, dtype=torch.long))
        self.register_buffer("mm", torch.tensor(data.mirna_gene_mask, dtype=torch.bool))
        self.register_buffer("mp", torch.tensor(data.mirna_gene_prior, dtype=torch.float32))
        self.register_buffer("di", torch.tensor(data.disease_gene_indices, dtype=torch.long))
        self.register_buffer("dm", torch.tensor(data.disease_gene_mask, dtype=torch.bool))
        self.register_buffer("dp", torch.tensor(data.disease_gene_prior, dtype=torch.float32))

        h, g = CONFIG.endpoint_hidden_dim, CONFIG.gene_hidden_dim
        self.rna_project = nn.Sequential(
            nn.Linear(self.rna.shape[1], 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(CONFIG.dropout), nn.Linear(256, h), nn.LayerNorm(h), nn.GELU(),
        )
        self.member_attention = nn.Sequential(nn.Linear(h, 64), nn.Tanh(), nn.Linear(64, 1))
        self.disease_project = nn.Sequential(
            nn.Linear(self.disease.shape[1], 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(CONFIG.dropout), nn.Linear(256, h), nn.LayerNorm(h), nn.GELU(),
        )
        self.gene_project = nn.Sequential(
            nn.Linear(self.gene_context.shape[1], g), nn.LayerNorm(g), nn.GELU(),
            nn.Dropout(CONFIG.dropout / 2),
        )
        self.m_query, self.d_query = nn.Linear(h, g), nn.Linear(h, g)
        self.m_prior_scale, self.d_prior_scale = nn.Parameter(torch.zeros(())), nn.Parameter(torch.zeros(()))
        self.gene_to_endpoint = nn.Linear(g, h)
        gate_input = h + g + 2
        self.m_gate = nn.Sequential(nn.Linear(gate_input, 96), nn.GELU(), nn.Linear(96, 1))
        self.d_gate = nn.Sequential(nn.Linear(gate_input, 96), nn.GELU(), nn.Linear(96, 1))
        nn.init.constant_(self.m_gate[-1].bias, CONFIG.endpoint_gene_gate_bias)
        nn.init.constant_(self.d_gate[-1].bias, CONFIG.endpoint_gene_gate_bias)
        self.cross_project = nn.Sequential(
            nn.LayerNorm(4 * g + 4), nn.Linear(4 * g + 4, CONFIG.gene_cross_hidden_dim),
            nn.GELU(), nn.Dropout(CONFIG.dropout),
        )
        pair_input = 4 * h + 4 * g + CONFIG.gene_cross_hidden_dim + 8
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(pair_input), nn.Linear(pair_input, 320), nn.GELU(),
            nn.Dropout(CONFIG.dropout), nn.Linear(320, CONFIG.branch_dim),
            nn.LayerNorm(CONFIG.branch_dim), nn.GELU(),
        )
        self.classifier = nn.Linear(CONFIG.branch_dim, 1)

    def _mirna_endpoint(self, index: torch.Tensor) -> torch.Tensor:
        group, mask = self.groups[index], self.group_mask[index]
        members = self.rna_project(self.rna[group.clamp_min(0)])
        score = self.member_attention(members).squeeze(-1).masked_fill(~mask, -1e4)
        weight = torch.softmax(score, dim=1) * mask.float()
        weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return F.normalize((weight.unsqueeze(-1) * members).sum(dim=1), dim=-1)

    def _endpoint_gene_pool(self, indices, mask, prior, endpoint, query, prior_scale):
        token = self.gene_project(self.gene_context[indices.clamp_min(0)])
        q = F.normalize(query(endpoint), dim=-1)
        score = (F.normalize(token, dim=-1) * q[:, None]).sum(-1) / math.sqrt(token.shape[-1])
        score = score + F.softplus(prior_scale) * torch.log(prior.clamp_min(1e-4))
        score = score.masked_fill(~mask, -1e4)
        weight = torch.softmax(score, dim=1) * mask.float()
        weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = F.normalize((weight[:, :, None] * token).sum(dim=1), dim=-1)
        available = mask.any(dim=1).float()
        return token, pooled * available[:, None], available, mask.sum(dim=1).float()

    def _gene_cross_interaction(self, mg, dg, mi, di, mm, dm, mp, dp):
        valid = mm[:, :, None] & dm[:, None, :]
        compatibility = torch.einsum(
            "bik,bjk->bij", F.normalize(mg, dim=-1), F.normalize(dg, dim=-1),
        )
        score = compatibility + 0.15 * torch.log(mp[:, :, None].clamp_min(1e-4))
        score = score + 0.15 * torch.log(dp[:, None, :].clamp_min(1e-4))
        score = score.masked_fill(~valid, -1e4)
        weight = torch.softmax(score.flatten(1), dim=1).reshape_as(score) * valid.float()
        weight = weight / weight.sum((1, 2), keepdim=True).clamp_min(1e-8)
        m_pool = torch.einsum("bij,bik->bk", weight, mg)
        d_pool = torch.einsum("bij,bjk->bk", weight, dg)
        available = valid.flatten(1).any(1).float()
        direct = valid & (mi[:, :, None] == di[:, None, :])
        shared_count = direct.flatten(1).sum(1).float()
        shared_strength = (
            direct.float() * torch.minimum(mp[:, :, None], dp[:, None, :])
        ).sum((1, 2))
        maximum = compatibility.masked_fill(~valid, -1.0).flatten(1).max(1).values
        scalars = torch.stack([
            available,
            torch.log1p(shared_count) / math.log(1.0 + min(mi.shape[1], di.shape[1])),
            shared_strength.clamp(0.0, 1.0),
            maximum * available,
        ], dim=1)
        cross = self.cross_project(torch.cat([
            m_pool, d_pool, m_pool * d_pool, torch.abs(m_pool - d_pool), scalars,
        ], dim=1))
        return cross * available[:, None], available, scalars

    def _fuse_endpoint(self, endpoint, gene, available, count, gate, denominator):
        effective = available
        if self.training and CONFIG.gene_modality_dropout > 0:
            effective = available * (torch.rand_like(available) >= CONFIG.gene_modality_dropout).float()
        count_feature = torch.log1p(count) / math.log1p(denominator)
        value = torch.sigmoid(gate(torch.cat([
            endpoint, gene, effective[:, None], count_feature[:, None],
        ], dim=1))) * effective[:, None]
        return F.normalize(endpoint + value * self.gene_to_endpoint(gene), dim=-1), value

    def forward(self, m: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        me = self._mirna_endpoint(m)
        de = F.normalize(self.disease_project(self.disease[d]), dim=-1)
        mi, mm, mp = self.mi[m], self.mm[m], self.mp[m]
        di, dm, dp = self.di[d], self.dm[d], self.dp[d]
        mg_tokens, mg, ma, mc = self._endpoint_gene_pool(mi, mm, mp, me, self.m_query, self.m_prior_scale)
        dg_tokens, dg, da, dc = self._endpoint_gene_pool(di, dm, dp, de, self.d_query, self.d_prior_scale)
        cross, cross_available, cross_scalars = self._gene_cross_interaction(
            mg_tokens, dg_tokens, mi, di, mm, dm, mp, dp,
        )
        me, m_gate = self._fuse_endpoint(me, mg, ma, mc, self.m_gate, mi.shape[1])
        de, d_gate = self._fuse_endpoint(de, dg, da, dc, self.d_gate, di.shape[1])
        both = ma * da
        scalars = torch.cat([
            (me * de).sum(1, keepdim=True),
            (mg * dg).sum(1, keepdim=True) * both[:, None],
            ma[:, None], da[:, None], m_gate, d_gate,
            cross_available[:, None], cross_scalars[:, 1:2],
        ], dim=1)
        pair = torch.cat([
            me, de, me * de, torch.abs(me - de),
            mg, dg, mg * dg, torch.abs(mg - dg), cross, scalars,
        ], dim=1)
        return self.classifier(self.pair_encoder(pair)).squeeze(1)


def make_loader(arrays, batch_size: int, shuffle: bool, seed: int):
    m, d, y = arrays
    dataset = TensorDataset(
        torch.as_tensor(m, dtype=torch.long),
        torch.as_tensor(d, dtype=torch.long),
        torch.as_tensor(y, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
        num_workers=0, pin_memory=DEVICE.type == "cuda",
    )


def make_balanced_bio_loader(arrays, batch_size: int, seed: int):
    if batch_size < 2 or batch_size % 2 != 0:
        raise ValueError("BIO batch_size must be an even integer >= 2.")
    m, d, y = arrays
    labels = np.asarray(y, dtype=np.int64)
    positive = np.flatnonzero(labels == 1).astype(np.int64)
    negative = np.flatnonzero(labels == 0).astype(np.int64)
    if not len(positive) or not len(negative):
        raise ValueError("Balanced BIO training requires both classes.")

    rng = np.random.default_rng(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)
    half = batch_size // 2
    batch_count = max(
        int(math.ceil(len(positive) / half)),
        int(math.ceil(len(negative) / half)),
    )
    batches = []
    for batch_id in range(batch_count):
        p_start = batch_id * half
        n_start = batch_id * half
        p = positive[p_start:p_start + half]
        n = negative[n_start:n_start + half]


        target = max(len(p), len(n))
        if target == 0:
            continue
        if len(p) < target:
            p = np.concatenate([p, rng.choice(positive, target - len(p), replace=True)])
        if len(n) < target:
            n = np.concatenate([n, rng.choice(negative, target - len(n), replace=True)])
        batch = np.concatenate([p, n]).astype(np.int64)
        rng.shuffle(batch)
        batches.append(batch.tolist())

    dataset = TensorDataset(
        torch.as_tensor(m, dtype=torch.long),
        torch.as_tensor(d, dtype=torch.long),
        torch.as_tensor(y, dtype=torch.float32),
    )
    return DataLoader(
        dataset,
        batch_sampler=batches,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )


def make_scaler():
    return torch.amp.GradScaler("cuda", enabled=bool(CONFIG.use_amp and DEVICE.type == "cuda"))


class ExponentialMovingAverage:

    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}.")
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.backup: Dict[str, torch.Tensor] = {}
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = dict(model.named_parameters())
        self.num_updates += 1
        for name, shadow in self.shadow.items():
            source = current[name].detach()
            shadow.mul_(self.decay).add_(source, alpha=1.0 - self.decay)

    @contextmanager
    def average_parameters(self, model: nn.Module):
        current = dict(model.named_parameters())
        self.backup = {}
        try:
            for name, shadow in self.shadow.items():
                parameter = current[name]
                self.backup[name] = parameter.detach().clone()
                parameter.copy_(shadow)
            yield
        finally:
            for name, value in self.backup.items():
                current[name].copy_(value)
            self.backup = {}

    def state_dict(self) -> Dict[str, object]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()},
        }


def bio_pairwise_auc_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positive = logits[labels > 0.5].float()
    negative = logits[labels <= 0.5].float()
    if positive.numel() == 0 or negative.numel() == 0:
        return logits.new_zeros(())
    difference = (
        positive[:, None] - negative[None, :]
    ) / CONFIG.bio_ranking_temperature
    return F.softplus(-difference).mean()


def train_bio_epoch(model, arrays, optimizer, scaler, ema: ExponentialMovingAverage, seed: int):
    model.train()
    total, bce_total, ranking_total, count = 0.0, 0.0, 0.0, 0
    for m, d, labels in make_balanced_bio_loader(arrays, CONFIG.batch_size, seed):
        m, d, labels = m.to(DEVICE), d.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16,
            enabled=bool(CONFIG.use_amp and DEVICE.type == "cuda"),
        ):
            logits = model(m, d)
            bce = F.binary_cross_entropy_with_logits(logits, labels)
            ranking = bio_pairwise_auc_loss(logits, labels)
            loss = (
                (1.0 - CONFIG.bio_ranking_weight) * bce
                + CONFIG.bio_ranking_weight * ranking
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        current_count = len(labels)
        total += float(loss.detach()) * current_count
        bce_total += float(bce.detach()) * current_count
        ranking_total += float(ranking.detach()) * current_count
        count += current_count
    denominator = max(count, 1)
    return {
        "loss": total / denominator,
        "bce_loss": bce_total / denominator,
        "ranking_loss": ranking_total / denominator,
    }


@torch.no_grad()
def predict_bio(model, arrays, ema: ExponentialMovingAverage | None = None) -> np.ndarray:
    def _predict() -> np.ndarray:
        model.eval()
        output = []
        for m, d, _ in make_loader(arrays, CONFIG.eval_batch_size, False, CONFIG.seed):
            output.append(torch.sigmoid(model(m.to(DEVICE), d.to(DEVICE))).cpu().numpy())
        return np.concatenate(output).astype(np.float32)

    if ema is None:
        return _predict()
    with ema.average_parameters(model):
        return _predict()


def select_bio_epoch(data: AssociationDataBundle, train_arrays, validation_arrays, seed: int):
    started = time.time()
    set_seed(seed)
    progress(
        f"Biological branch selection: train={len(train_arrays[2])}, validation={len(validation_arrays[2])}, "
        f"maximum_epochs={CONFIG.maximum_epochs}"
    )
    model = GeneAnchoredBiologicalBranch(data).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG.bio_learning_rate, weight_decay=CONFIG.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.4, patience=3, min_lr=2e-6,
    )
    scaler = make_scaler()
    ema = ExponentialMovingAverage(model, CONFIG.bio_ema_decay)
    best_auc, patience_auc, best_epoch, stale = -1.0, -1.0, 1, 0
    history = []
    best_probability = None
    for epoch in range(1, CONFIG.maximum_epochs + 1):
        epoch_started = time.time()
        losses = train_bio_epoch(model, train_arrays, optimizer, scaler, ema, seed + epoch)
        probability = predict_bio(model, validation_arrays, ema)
        auc = safe_auc(validation_arrays[2], probability)
        scheduler.step(auc)
        history.append({
            "epoch": epoch,
            **losses,
            "val_auc": auc,
            "lr": optimizer.param_groups[0]["lr"],
            "ema_updates": ema.num_updates,
        })


        if auc > best_auc + 1e-10:
            best_auc, best_epoch = auc, epoch
            best_probability = probability.copy()
        if auc > patience_auc + CONFIG.bio_early_stop_min_delta:
            patience_auc, stale = auc, 0
        else:
            stale += 1
        progress(
            f"Biological selection epoch {epoch:03d}/{CONFIG.maximum_epochs}: "
            f"loss={losses['loss']:.6f}, bce={losses['bce_loss']:.6f}, "
            f"rank={losses['ranking_loss']:.6f}, val_auc={auc:.6f}, "
            f"best_auc={best_auc:.6f}@{best_epoch}, lr={optimizer.param_groups[0]['lr']:.2e}, "
            f"stale={stale}/{CONFIG.patience}, time={format_duration(time.time() - epoch_started)}"
        )
        if stale >= CONFIG.patience:
            progress(
                f"Biological branch selection: early stopping at epoch {epoch}; selected epoch={best_epoch}"
            )
            break
    if best_probability is None:
        raise RuntimeError("BIO validation prediction was not captured.")
    progress(
        f"Biological branch selection completed: selected_epoch={best_epoch}, "
        f"best_val_auc={best_auc:.6f}, elapsed={format_duration(time.time() - started)}"
    )
    del model, ema
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return int(best_epoch), float(best_auc), history, best_probability


def fit_bio_from_scratch(data: AssociationDataBundle, arrays, epochs: int, seed: int):
    started = time.time()
    set_seed(seed)
    progress(f"Biological branch refit: samples={len(arrays[2])}, epochs={epochs}")
    model = GeneAnchoredBiologicalBranch(data).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG.bio_learning_rate, weight_decay=CONFIG.weight_decay,
    )
    scaler = make_scaler()
    ema = ExponentialMovingAverage(model, CONFIG.bio_ema_decay)
    history = []
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        losses = train_bio_epoch(model, arrays, optimizer, scaler, ema, seed + epoch)
        history.append({"epoch": epoch, **losses, "ema_updates": ema.num_updates})
        progress(
            f"Biological refit epoch {epoch:03d}/{epochs}: loss={losses['loss']:.6f}, "
            f"bce={losses['bce_loss']:.6f}, rank={losses['ranking_loss']:.6f}, "
            f"ema_updates={ema.num_updates}, time={format_duration(time.time() - epoch_started)}"
        )
    model.eval()
    progress(f"Biological branch refit completed in {format_duration(time.time() - started)}")
    return model, ema, history


def calculate_gip_kernel(profiles: np.ndarray) -> np.ndarray:
    value = np.asarray(profiles, dtype=np.float32)
    squared_norm = np.sum(value * value, axis=1, dtype=np.float64)
    mean_squared_norm = float(np.mean(squared_norm))
    gamma = 1.0 if mean_squared_norm <= 1e-12 else 1.0 / mean_squared_norm
    distance = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (value @ value.T)
    kernel = np.exp(-gamma * np.maximum(distance, 0.0)).astype(np.float32)
    kernel = np.clip(np.nan_to_num(kernel, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    np.fill_diagonal(kernel, 1.0)
    return kernel


def calculate_fold_gip(association: np.ndarray):
    return calculate_gip_kernel(association), calculate_gip_kernel(association.T)


def calculate_mirna_functional_similarity(
    association: np.ndarray,
    disease_semantic_similarity: np.ndarray,
) -> np.ndarray:
    edge_matrix = (np.asarray(association) > 0).astype(np.float32)
    semantic = np.asarray(disease_semantic_similarity, dtype=np.float32)
    mirna_count, disease_count = edge_matrix.shape
    if semantic.shape != (disease_count, disease_count):
        raise ValueError(
            "Disease semantic similarity shape does not match the association matrix: "
            f"expected {(disease_count, disease_count)}, got {semantic.shape}."
        )


    best_to_set = np.zeros((disease_count, mirna_count), dtype=np.float32)
    for mirna_index in range(mirna_count):
        related = np.flatnonzero(edge_matrix[mirna_index] > 0.0)
        if related.size:
            best_to_set[:, mirna_index] = np.max(semantic[:, related], axis=1)


    directed = edge_matrix @ best_to_set
    degree = edge_matrix.sum(axis=1, dtype=np.float32)
    denominator = degree[:, None] + degree[None, :]
    similarity = np.divide(
        directed + directed.T,
        denominator,
        out=np.zeros((mirna_count, mirna_count), dtype=np.float32),
        where=denominator > 0.0,
    )
    similarity = np.clip(
        np.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0,
    )
    similarity = 0.5 * (similarity + similarity.T)
    np.fill_diagonal(similarity, 1.0)
    return similarity.astype(np.float32)


def sparsify_similarity_topk(matrix: np.ndarray, k: int) -> np.ndarray:
    similarity = np.asarray(matrix, dtype=np.float32).copy()
    similarity = np.clip(np.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    np.fill_diagonal(similarity, 0.0)
    n = similarity.shape[0]
    k = max(1, min(int(k), n - 1))
    indices = np.argpartition(similarity, kth=n - k, axis=1)[:, -k:]
    retained = np.zeros_like(similarity)
    rows = np.arange(n)[:, None]
    retained[rows, indices] = similarity[rows, indices]
    return np.maximum(retained, retained.T)


def build_normalized_adj(matrix: np.ndarray) -> torch.Tensor:
    adjacency = np.asarray(matrix, dtype=np.float32).copy()
    adjacency[adjacency < 0] = 0.0
    adjacency += np.eye(adjacency.shape[0], dtype=np.float32)
    degree = adjacency.sum(axis=1)
    inverse = np.power(degree + 1e-8, -0.5)
    return torch.tensor((adjacency * inverse[:, None]) * inverse[None, :], dtype=torch.float32)


def normalize_bipartite(association: torch.Tensor) -> torch.Tensor:
    row_degree = association.sum(dim=1).clamp_min(1e-8)
    column_degree = association.sum(dim=0).clamp_min(1e-8)
    return association * torch.rsqrt(row_degree)[:, None] * torch.rsqrt(column_degree)[None, :]


def log_standardize(value: torch.Tensor) -> torch.Tensor:
    value = torch.log1p(value.clamp_min(0.0))
    return (value - value.mean()) / value.std(unbiased=False).clamp_min(1e-6)


def build_structural_channels(association, mirna_similarity, disease_similarity, mirna_gip, disease_gip):
    a = torch.tensor(np.asarray(association, dtype=np.float32))
    mirna_degree, disease_degree = a.sum(dim=1), a.sum(dim=0)
    b = normalize_bipartite(a)
    path3_normalized = (b @ b.T) @ b
    path5_normalized = (path3_normalized @ b.T) @ b
    path3_raw = (a @ a.T) @ a
    path3_ra = ((a / disease_degree.clamp_min(1e-6)[None, :]) @ a.T) @ (
        a / mirna_degree.clamp_min(1e-6)[:, None]
    )
    static_m = build_normalized_adj(sparsify_similarity_topk(mirna_similarity, CONFIG.similarity_topk))
    static_d = build_normalized_adj(sparsify_similarity_topk(disease_similarity, CONFIG.similarity_topk))
    gip_m = build_normalized_adj(sparsify_similarity_topk(mirna_gip, CONFIG.similarity_topk))
    gip_d = build_normalized_adj(sparsify_similarity_topk(disease_gip, CONFIG.similarity_topk))
    static_left, static_right = static_m @ b, b @ static_d
    gip_left, gip_right = gip_m @ b, b @ gip_d
    channels = torch.stack([
        log_standardize(path3_normalized), log_standardize(path5_normalized),
        log_standardize(path3_raw), log_standardize(path3_ra),
        log_standardize(static_left), log_standardize(static_right),
        log_standardize(static_left @ static_d), log_standardize(gip_left),
        log_standardize(gip_right), log_standardize(gip_left @ gip_d),
    ], dim=0).float()
    return channels, log_standardize(mirna_degree).float(), log_standardize(disease_degree).float()


def select_structural_features(bank, mirna, disease, overlap_features):
    channels, mirna_degree, disease_degree = bank
    m = torch.as_tensor(mirna, dtype=torch.long)
    d = torch.as_tensor(disease, dtype=torch.long)
    overlap = torch.as_tensor(
        overlap_features[np.asarray(mirna, dtype=np.int64), np.asarray(disease, dtype=np.int64)],
        dtype=torch.float32,
    )
    result = torch.cat([
        channels[:, m, d].T, mirna_degree[m, None], disease_degree[d, None], overlap,
    ], dim=1).numpy().astype(np.float32, copy=False)
    if result.shape[1] != 15 or not np.isfinite(result).all():
        raise RuntimeError("Structural feature construction failed its 15-channel audit.")
    return result


@dataclass
class StructuralStageData:
    association: np.ndarray
    association_views: List[np.ndarray]
    mirna_gip: np.ndarray
    disease_gip: np.ndarray
    mirna_gip_views: List[np.ndarray]
    disease_gip_views: List[np.ndarray]
    graph_ids: np.ndarray


def build_train_association(shape, arrays):
    m, d, y = arrays
    result = np.zeros(shape, dtype=np.float32)
    positive = np.asarray(y) == 1
    result[np.asarray(m)[positive], np.asarray(d)[positive]] = 1.0
    return result


def build_stage_data(shape, arrays, seed: int):
    m, d, y = arrays
    association = build_train_association(shape, arrays)
    positive = np.flatnonzero(np.asarray(y) == 1)
    negative = np.flatnonzero(np.asarray(y) == 0)
    if len(positive) < CONFIG.crossfit_graph_views:
        raise ValueError("Not enough positives for structural cross-fitting.")
    rng = np.random.default_rng(seed)
    chunks = np.array_split(rng.permutation(positive), CONFIG.crossfit_graph_views)
    graph_ids = np.empty(len(y), dtype=np.int64)
    for graph_id, chunk in enumerate(chunks):
        graph_ids[chunk] = graph_id
    graph_ids[negative] = rng.integers(0, CONFIG.crossfit_graph_views, size=len(negative))
    association_views, mirna_views, disease_views = [], [], []
    for chunk in chunks:
        view = association.copy()
        view[np.asarray(m)[chunk], np.asarray(d)[chunk]] = 0.0
        mirna_gip, disease_gip = calculate_fold_gip(view)
        association_views.append(view)
        mirna_views.append(mirna_gip)
        disease_views.append(disease_gip)
    mirna_gip, disease_gip = calculate_fold_gip(association)
    return StructuralStageData(
        association, association_views, mirna_gip, disease_gip,
        mirna_views, disease_views, graph_ids,
    )


def fit_structural_expert(stage, arrays, data: AssociationDataBundle, seed: int):
    started = time.time()
    m, d, y = arrays
    features = np.empty((len(y), 15), dtype=np.float32)
    progress(f"Structural branch: building cross-fitted features for {len(y)} samples")
    for graph_id in range(CONFIG.crossfit_graph_views):
        selected = np.flatnonzero(stage.graph_ids == graph_id)
        if not len(selected):
            progress(
                f"Structural branch: graph view {graph_id + 1}/{CONFIG.crossfit_graph_views} has no samples; skipped"
            )
            continue
        progress(
            f"Structural branch: graph view {graph_id + 1}/{CONFIG.crossfit_graph_views}, samples={len(selected)}"
        )
        positive = selected[np.asarray(y)[selected] == 1]
        if len(positive) and np.any(stage.association_views[graph_id][m[positive], d[positive]] != 0):
            raise RuntimeError("A held positive edge remained in a structural training view.")
        bank = build_structural_channels(
            stage.association_views[graph_id], data.mirna_similarity, data.disease_similarity,
            stage.mirna_gip_views[graph_id], stage.disease_gip_views[graph_id],
        )
        features[selected] = select_structural_features(
            bank, m[selected], d[selected], data.overlap_features,
        )
    progress(
        f"Structural branch: fitting HistGradientBoostingClassifier, max_iter={CONFIG.expert_max_iter}"
    )
    expert = HistGradientBoostingClassifier(
        max_iter=CONFIG.expert_max_iter,
        learning_rate=CONFIG.expert_learning_rate,
        max_leaf_nodes=CONFIG.expert_max_leaf_nodes,
        min_samples_leaf=CONFIG.expert_min_samples_leaf,
        l2_regularization=CONFIG.expert_l2_regularization,
        random_state=seed,
    )
    expert.fit(features, np.asarray(y, dtype=np.int64))
    progress(f"Structural branch: fit completed in {format_duration(time.time() - started)}")
    return expert


def predict_structural_expert(expert, stage, arrays, data: AssociationDataBundle):
    m, d, _ = arrays
    bank = build_structural_channels(
        stage.association, data.mirna_similarity, data.disease_similarity,
        stage.mirna_gip, stage.disease_gip,
    )
    features = select_structural_features(bank, m, d, data.overlap_features)
    return expert.predict_proba(features)[:, 1].astype(np.float32)


@dataclass
class CompletionFeatureData:
    score_bank: np.ndarray
    mirna_degree: np.ndarray
    disease_degree: np.ndarray


@dataclass
class CompletionStageData:
    score_bank: np.ndarray
    mirna_degree: np.ndarray
    disease_degree: np.ndarray
    graph_ids: np.ndarray
    full_view_id: int
    feature_names: List[str]


def row_normalize_numpy(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    denominator = value.sum(axis=1, keepdims=True)
    return value / np.maximum(denominator, 1e-8)


def standardize_score_matrix(matrix: np.ndarray) -> np.ndarray:
    value = np.nan_to_num(
        np.asarray(matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0,
    )
    mean = float(value.mean())
    standard_deviation = float(value.std())
    return ((value - mean) / max(standard_deviation, 1e-6)).astype(np.float32)


def standardize_score_vector(vector: np.ndarray) -> np.ndarray:
    value = np.nan_to_num(
        np.asarray(vector, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0,
    )
    return ((value - float(value.mean())) / max(float(value.std()), 1e-6)).astype(np.float32)


def row_stochastic_similarity(similarity: np.ndarray) -> np.ndarray:
    sparse = sparsify_similarity_topk(similarity, CONFIG.similarity_topk)
    sparse = sparse + np.eye(sparse.shape[0], dtype=np.float32)
    return row_normalize_numpy(sparse).astype(np.float32)


def truncated_svd_score_views(
    matrix: np.ndarray,
    ranks: Sequence[int],
    seed: int,
) -> List[np.ndarray]:
    value = np.asarray(matrix, dtype=np.float32)
    maximum_rank = max(1, min(value.shape) - 1)
    requested_rank = max(1, min(max(int(rank) for rank in ranks), maximum_rank))
    left, singular, right_t = randomized_svd(
        value,
        n_components=requested_rank,
        n_iter=7,
        random_state=seed,
    )
    left = left.astype(np.float32)
    singular = singular.astype(np.float32)
    right_t = right_t.astype(np.float32)
    views: List[np.ndarray] = []
    for requested in ranks:
        cutoff = max(1, min(int(requested), requested_rank))
        reconstruction = (
            (left[:, :cutoff] * singular[None, :cutoff]) @ right_t[:cutoff]
        )
        views.append(standardize_score_matrix(reconstruction))
    return views


def build_completion_feature_context(
    association: np.ndarray,
    data: AssociationDataBundle,
    seed: int,
) -> Tuple[CompletionFeatureData, List[str]]:
    edge_matrix = np.asarray(association, dtype=np.float32)
    row_degree = edge_matrix.sum(axis=1).astype(np.float32)
    column_degree = edge_matrix.sum(axis=0).astype(np.float32)

    normalized = edge_matrix / np.sqrt(np.maximum(row_degree[:, None], 1.0))
    normalized = normalized / np.sqrt(np.maximum(column_degree[None, :], 1.0))

    score_channels: List[np.ndarray] = []
    feature_names: List[str] = []
    for name, matrix, offset in [
        ("raw", edge_matrix, 0),
        ("degree_normalized", normalized, 101),
    ]:
        views = truncated_svd_score_views(
            matrix, CONFIG.completion_score_ranks, seed + offset,
        )
        score_channels.extend(views)
        feature_names.extend([
            f"{name}_svd_rank_{int(rank)}" for rank in CONFIG.completion_score_ranks
        ])

    mirna_gip, disease_gip = calculate_fold_gip(edge_matrix)


    sequence_m = row_stochastic_similarity(data.mirna_similarity)
    semantic_d = row_stochastic_similarity(data.disease_similarity)


    functional_similarity = calculate_mirna_functional_similarity(
        edge_matrix, data.disease_similarity,
    )
    functional_m = row_stochastic_similarity(functional_similarity)
    feature_d = row_stochastic_similarity(data.disease_feature_similarity)


    gip_m = row_stochastic_similarity(mirna_gip)
    gip_d = row_stochastic_similarity(disease_gip)


    mixed_m = row_normalize_numpy(0.55 * sequence_m + 0.45 * gip_m)
    mixed_d = row_normalize_numpy(0.55 * semantic_d + 0.45 * gip_d)


    enhanced_m = row_normalize_numpy(
        0.35 * sequence_m + 0.35 * functional_m + 0.30 * gip_m
    )
    enhanced_d = row_normalize_numpy(
        0.35 * semantic_d + 0.35 * feature_d + 0.30 * gip_d
    )

    propagation_channels = [

        ("static_left", sequence_m @ edge_matrix),
        ("static_right", edge_matrix @ semantic_d.T),
        ("static_both", (sequence_m @ edge_matrix) @ semantic_d.T),
        ("gip_left", gip_m @ edge_matrix),
        ("gip_right", edge_matrix @ gip_d.T),
        ("gip_both", (gip_m @ edge_matrix) @ gip_d.T),
        ("mixed_both", (mixed_m @ edge_matrix) @ mixed_d.T),


        ("functional_left", functional_m @ edge_matrix),
        ("feature_right", edge_matrix @ feature_d.T),
        ("functional_feature_both", (functional_m @ edge_matrix) @ feature_d.T),
        ("functional_semantic_both", (functional_m @ edge_matrix) @ semantic_d.T),
        ("sequence_feature_both", (sequence_m @ edge_matrix) @ feature_d.T),
        ("enhanced_multiview_both", (enhanced_m @ edge_matrix) @ enhanced_d.T),
    ]
    for name, matrix in propagation_channels:
        score_channels.append(standardize_score_matrix(matrix))
        feature_names.append(name)

    score_bank = np.stack(score_channels, axis=-1).astype(np.float32)
    expected_channel_count = 2 * len(CONFIG.completion_score_ranks) + 13
    if score_bank.shape[-1] != expected_channel_count:
        raise RuntimeError(
            f"D25 completion expected {expected_channel_count} channels, "
            f"got {score_bank.shape[-1]}."
        )
    if not np.isfinite(score_bank).all():
        raise RuntimeError("Completion feature context contains non-finite values.")

    mirna_degree = standardize_score_vector(np.log1p(row_degree))
    disease_degree = standardize_score_vector(np.log1p(column_degree))
    return CompletionFeatureData(
        score_bank=score_bank,
        mirna_degree=mirna_degree,
        disease_degree=disease_degree,
    ), feature_names


def build_completion_stage(
    shape,
    arrays,
    data: AssociationDataBundle,
    seed: int,
) -> CompletionStageData:
    mirna, disease, labels = arrays
    mirna = np.asarray(mirna, dtype=np.int64)
    disease = np.asarray(disease, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    association = build_train_association(shape, arrays)

    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if len(positive) < CONFIG.crossfit_graph_views:
        raise ValueError("Not enough positives for completion held-edge cross-fitting.")

    rng = np.random.default_rng(seed)
    positive_chunks = np.array_split(
        rng.permutation(positive), CONFIG.crossfit_graph_views,
    )
    graph_ids = np.empty(len(labels), dtype=np.int64)
    contexts: List[CompletionFeatureData] = []
    feature_names: List[str] | None = None

    stage_started = time.time()
    progress(
        f"Completion branch: building {CONFIG.crossfit_graph_views} held-edge feature views"
    )
    for graph_id, chunk in enumerate(positive_chunks):
        view_started = time.time()
        progress(
            f"Completion branch: held-edge view {graph_id + 1}/{CONFIG.crossfit_graph_views}, held positives={len(chunk)}"
        )
        graph_ids[chunk] = graph_id
        held_view = association.copy()
        held_view[mirna[chunk], disease[chunk]] = 0.0
        if len(chunk) and np.any(held_view[mirna[chunk], disease[chunk]] != 0.0):
            raise RuntimeError("A held positive remained in a completion training view.")
        context, names = build_completion_feature_context(
            held_view, data, seed + graph_id * 211 + 17,
        )
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("Completion feature channel order changed across views.")
        contexts.append(context)
        progress(
            f"Completion branch: held-edge view {graph_id + 1}/{CONFIG.crossfit_graph_views} ready "
            f"({format_duration(time.time() - view_started)})"
        )

    graph_ids[negative] = rng.integers(
        0, CONFIG.crossfit_graph_views, size=len(negative),
    )
    progress("Completion branch: building full inference feature view")
    full_context, names = build_completion_feature_context(
        association, data, seed + 9001,
    )
    if feature_names is None:
        feature_names = names
    elif feature_names != names:
        raise RuntimeError("Completion full-view feature order differs from held views.")
    contexts.append(full_context)
    progress(
        f"Completion branch: feature bank ready in {format_duration(time.time() - stage_started)}"
    )

    score_bank = np.stack([context.score_bank for context in contexts], axis=0)
    mirna_degree = np.stack([context.mirna_degree for context in contexts], axis=0)
    disease_degree = np.stack([context.disease_degree for context in contexts], axis=0)
    if score_bank.shape[0] != CONFIG.crossfit_graph_views + 1:
        raise RuntimeError("Completion context-bank view count is incorrect.")
    return CompletionStageData(
        score_bank=score_bank.astype(np.float32),
        mirna_degree=mirna_degree.astype(np.float32),
        disease_degree=disease_degree.astype(np.float32),
        graph_ids=graph_ids,
        full_view_id=CONFIG.crossfit_graph_views,
        feature_names=list(feature_names),
    )


class CrossFittedCompletionBranch(nn.Module):
    def __init__(self, stage: CompletionStageData):
        super().__init__()
        self.full_view_id = int(stage.full_view_id)
        self.feature_names = list(stage.feature_names)
        self.register_buffer(
            "score_bank", torch.tensor(stage.score_bank, dtype=torch.float32),
        )
        self.register_buffer(
            "mirna_degree", torch.tensor(stage.mirna_degree, dtype=torch.float32),
        )
        self.register_buffer(
            "disease_degree", torch.tensor(stage.disease_degree, dtype=torch.float32),
        )
        score_count = int(stage.score_bank.shape[-1])
        pair_dimension = score_count + 8
        self.score_head = nn.Linear(score_count, 1, bias=False)
        with torch.no_grad():
            initial = torch.zeros((1, score_count), dtype=torch.float32)
            svd_count = 2 * len(CONFIG.completion_score_ranks)
            initial[:, :svd_count] = 1.0 / max(svd_count, 1)
            self.score_head.weight.copy_(initial)
        self.residual = nn.Sequential(
            nn.LayerNorm(pair_dimension),
            nn.Linear(pair_dimension, CONFIG.completion_hidden_dim),
            nn.GELU(),
            nn.Dropout(CONFIG.completion_dropout),
            nn.Linear(CONFIG.completion_hidden_dim, 1),
        )
        self.global_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        mirna: torch.Tensor,
        disease: torch.Tensor,
        graph_id: torch.Tensor,
    ) -> torch.Tensor:
        scores = self.score_bank[graph_id, mirna, disease]
        mirna_degree = self.mirna_degree[graph_id, mirna]
        disease_degree = self.disease_degree[graph_id, disease]
        degree = torch.stack([mirna_degree, disease_degree], dim=1)
        score_statistics = torch.stack([
            scores.mean(dim=1),
            scores.std(dim=1, unbiased=False),
            scores.max(dim=1).values,
            scores.min(dim=1).values,
        ], dim=1)
        pair = torch.cat([
            scores,
            degree,
            (mirna_degree * disease_degree)[:, None],
            torch.abs(mirna_degree - disease_degree)[:, None],
            score_statistics,
        ], dim=1)
        linear_logit = self.score_head(scores).squeeze(1)
        residual_logit = self.residual(pair).squeeze(1)
        return (
            linear_logit
            + CONFIG.completion_residual_scale * residual_logit
            + self.global_bias
        )


def make_completion_loader(
    arrays,
    graph_ids: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    mirna, disease, labels = arrays
    dataset = TensorDataset(
        torch.as_tensor(mirna, dtype=torch.long),
        torch.as_tensor(disease, dtype=torch.long),
        torch.as_tensor(labels, dtype=torch.float32),
        torch.as_tensor(graph_ids, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )


def completion_pairwise_auc_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    positive = logits[labels > 0.5]
    negative = logits[labels <= 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        return logits.new_zeros(())
    difference = (
        positive[:, None] - negative[None, :]
    ) / CONFIG.completion_ranking_temperature
    return F.softplus(-difference).mean()


def train_completion_epoch(
    model,
    arrays,
    graph_ids: np.ndarray,
    optimizer,
    seed: int,
) -> float:
    model.train()
    total, count = 0.0, 0
    for mirna, disease, labels, view in make_completion_loader(
        arrays, graph_ids, CONFIG.completion_batch_size, True, seed,
    ):
        mirna = mirna.to(DEVICE)
        disease = disease.to(DEVICE)
        labels = labels.to(DEVICE)
        view = view.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(mirna, disease, view)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        ranking = completion_pairwise_auc_loss(logits, labels)
        loss = (
            (1.0 - CONFIG.completion_ranking_weight) * bce
            + CONFIG.completion_ranking_weight * ranking
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), CONFIG.completion_gradient_clip,
        )
        optimizer.step()
        total += float(loss.detach()) * len(labels)
        count += len(labels)
    return total / max(count, 1)


@torch.no_grad()
def predict_completion(
    model,
    arrays,
    graph_ids: np.ndarray | None = None,
) -> np.ndarray:
    model.eval()
    if graph_ids is None:
        graph_ids = np.full(
            len(arrays[2]), model.full_view_id, dtype=np.int64,
        )
    output = []
    for mirna, disease, _, view in make_completion_loader(
        arrays, graph_ids, CONFIG.completion_eval_batch_size, False, CONFIG.seed,
    ):
        probability = torch.sigmoid(model(
            mirna.to(DEVICE), disease.to(DEVICE), view.to(DEVICE),
        ))
        output.append(probability.cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def select_completion_epoch(data: AssociationDataBundle, train_arrays, validation_arrays, seed: int):
    started = time.time()
    set_seed(seed)
    progress(
        f"Completion branch selection: train={len(train_arrays[2])}, validation={len(validation_arrays[2])}, "
        f"maximum_epochs={CONFIG.completion_maximum_epochs}"
    )
    stage = build_completion_stage(
        data.association.shape, train_arrays, data, seed,
    )
    model = CrossFittedCompletionBranch(stage).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG.completion_learning_rate,
        weight_decay=CONFIG.completion_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5,
    )
    best_auc, patience_auc, best_epoch, stale = -1.0, -1.0, 1, 0
    history = []
    best_probability = None
    for epoch in range(1, CONFIG.completion_maximum_epochs + 1):
        epoch_started = time.time()
        loss = train_completion_epoch(
            model, train_arrays, stage.graph_ids, optimizer, seed + epoch,
        )
        probability = predict_completion(model, validation_arrays)
        auc = safe_auc(validation_arrays[2], probability)
        scheduler.step(auc)
        history.append({
            "epoch": epoch,
            "loss": loss,
            "val_auc": auc,
            "lr": optimizer.param_groups[0]["lr"],
        })


        if auc > best_auc + 1e-10:
            best_auc, best_epoch = auc, epoch
            best_probability = probability.copy()
        if auc > patience_auc + CONFIG.completion_early_stop_min_delta:
            patience_auc, stale = auc, 0
        else:
            stale += 1
        progress(
            f"Completion selection epoch {epoch:03d}/{CONFIG.completion_maximum_epochs}: "
            f"loss={loss:.6f}, val_auc={auc:.6f}, best_auc={best_auc:.6f}@{best_epoch}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}, stale={stale}/{CONFIG.completion_patience}, "
            f"time={format_duration(time.time() - epoch_started)}"
        )
        if stale >= CONFIG.completion_patience:
            progress(
                f"Completion branch selection: early stopping at epoch {epoch}; selected epoch={best_epoch}"
            )
            break
    if best_probability is None:
        raise RuntimeError("Completion validation prediction was not captured.")
    progress(
        f"Completion branch selection completed: selected_epoch={best_epoch}, "
        f"best_val_auc={best_auc:.6f}, elapsed={format_duration(time.time() - started)}"
    )
    del model, stage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return int(best_epoch), float(best_auc), history, best_probability


def fit_completion_from_scratch(data: AssociationDataBundle, arrays, epochs: int, seed: int):
    started = time.time()
    set_seed(seed)
    progress(f"Completion branch refit: samples={len(arrays[2])}, epochs={epochs}")
    stage = build_completion_stage(data.association.shape, arrays, data, seed)
    model = CrossFittedCompletionBranch(stage).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG.completion_learning_rate,
        weight_decay=CONFIG.completion_weight_decay,
    )
    history = []
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        loss = train_completion_epoch(
            model, arrays, stage.graph_ids, optimizer, seed + epoch,
        )
        history.append({"epoch": epoch, "loss": loss})
        progress(
            f"Completion refit epoch {epoch:03d}/{epochs}: loss={loss:.6f}, "
            f"time={format_duration(time.time() - epoch_started)}"
        )
    model.eval()
    del stage
    progress(f"Completion branch refit completed in {format_duration(time.time() - started)}")
    return model, history


def sample_random_balanced_pairs(association: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    flat = association.reshape(-1)
    positive = np.flatnonzero(flat > 0)
    unknown = np.flatnonzero(flat == 0)
    negative = rng.choice(unknown, size=len(positive), replace=False)
    selected = np.concatenate([positive, negative])
    labels = np.concatenate([
        np.ones(len(positive), dtype=np.int64), np.zeros(len(negative), dtype=np.int64),
    ])
    order = rng.permutation(len(selected))
    selected, labels = selected[order], labels[order]
    mirna, disease = np.unravel_index(selected, association.shape)
    return mirna.astype(np.int64), disease.astype(np.int64), labels


def make_balanced_folds(labels: np.ndarray, fold_count: int, seed: int):
    positive = np.flatnonzero(labels == 1).astype(np.int64)
    negative = np.flatnonzero(labels == 0).astype(np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(positive); rng.shuffle(negative)
    positive_chunks, negative_chunks = np.array_split(positive, fold_count), np.array_split(negative, fold_count)
    all_indices = np.arange(len(labels), dtype=np.int64)
    folds = []
    for fold, (pos_test, neg_test) in enumerate(zip(positive_chunks, negative_chunks), start=1):
        test = np.concatenate([pos_test, neg_test]).astype(np.int64)
        fold_rng = np.random.default_rng(seed + fold * 1543)
        fold_rng.shuffle(test)
        mask = np.ones(len(labels), dtype=np.bool_); mask[test] = False
        train = all_indices[mask]; fold_rng.shuffle(train)
        folds.append((train, test))
    return folds


def make_inner_split(labels: np.ndarray, seed: int):
    indices = np.arange(len(labels), dtype=np.int64)
    train, validation = train_test_split(
        indices, test_size=CONFIG.inner_validation_ratio,
        random_state=seed, stratify=labels,
    )
    return train.astype(np.int64), validation.astype(np.int64)


def rank_normalize(probability: np.ndarray) -> np.ndarray:
    value = np.asarray(probability, dtype=np.float64).reshape(-1)
    order = np.argsort(value, kind="mergesort")
    ranks = np.empty(len(value), dtype=np.float64)
    ranks[order] = np.arange(len(value), dtype=np.float64)
    return ((ranks + 0.5) / max(len(value), 1)).astype(np.float32)


def rank_fuse_three(bio_probability, structure_probability, completion_probability, weights):
    bio_rank = rank_normalize(bio_probability)
    structure_rank = rank_normalize(structure_probability)
    completion_rank = rank_normalize(completion_probability)
    weight = np.asarray(weights, dtype=np.float32)
    if weight.shape != (3,) or not np.isclose(weight.sum(), 1.0):
        raise ValueError(f"Expected three fusion weights summing to one, got {weight}.")
    return (
        weight[0] * bio_rank + weight[1] * structure_rank + weight[2] * completion_rank
    ).astype(np.float32)


def select_fusion_weights(labels, bio_probability, structure_probability, completion_probability):
    labels = np.asarray(labels, dtype=np.int64)
    ranked = [
        rank_normalize(bio_probability),
        rank_normalize(structure_probability),
        rank_normalize(completion_probability),
    ]
    step = float(CONFIG.fusion_grid_step)
    units = int(round(1.0 / step))
    prior = np.asarray([0.10, 0.70, 0.20], dtype=np.float64)
    best_key, best_weight, best_probability = None, None, None
    for a_unit in range(units + 1):
        for b_unit in range(units - a_unit + 1):
            c_unit = units - a_unit - b_unit
            weight = np.asarray([a_unit, b_unit, c_unit], dtype=np.float64) / units
            if not (CONFIG.fusion_min_bio_weight <= weight[0] <= CONFIG.fusion_max_bio_weight):
                continue
            if weight[1] < CONFIG.fusion_min_structure_weight:
                continue
            if not (CONFIG.fusion_min_completion_weight <= weight[2] <= CONFIG.fusion_max_completion_weight):
                continue
            probability = sum(float(w) * p for w, p in zip(weight, ranked))
            auc = safe_auc(labels, probability)
            aupr = float(average_precision_score(labels, probability))
            distance = float(np.square(weight - prior).sum())
            key = (round(auc, 10), round(aupr, 10), -distance)
            if best_key is None or key > best_key:
                best_key, best_weight, best_probability = key, weight, probability
    if best_weight is None:
        raise RuntimeError("Fusion search constraints yielded no candidate weights.")
    return best_weight.astype(np.float32), float(best_key[0]), best_probability.astype(np.float32)


METRIC_NAMES = ("AUC", "AUPR", "ACCURACY", "PRECISION", "RECALL", "F1")


def summarize_main_model_metrics(rows):
    summary = {}
    for metric in METRIC_NAMES:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std())
    return summary


def build_mean_curve_data(curves):
    fpr_grid = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
    interpolated_tpr = []
    for curve in curves:
        current = np.interp(fpr_grid, curve["fpr"], curve["tpr"])
        current[0] = 0.0
        current[-1] = 1.0
        interpolated_tpr.append(current)
    mean_tpr = np.mean(interpolated_tpr, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0

    recall_grid = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
    interpolated_precision = []
    for curve in curves:
        recall = np.asarray(curve["recall"], dtype=np.float64)
        precision = np.asarray(curve["precision"], dtype=np.float64)
        order = np.argsort(recall, kind="mergesort")
        recall_sorted = recall[order]
        precision_sorted = precision[order]
        unique_recall, unique_indices = np.unique(recall_sorted, return_index=True)
        unique_precision = precision_sorted[unique_indices]
        interpolated_precision.append(
            np.interp(recall_grid, unique_recall, unique_precision)
        )
    mean_precision = np.mean(interpolated_precision, axis=0)
    return fpr_grid, mean_tpr, recall_grid, mean_precision


def save_curve_data(output_dir: Path, curves):
    fpr_grid, mean_tpr, recall_grid, mean_precision = build_mean_curve_data(curves)
    roc_data = {
        "mean_fpr": fpr_grid.astype(np.float32),
        "mean_tpr": mean_tpr.astype(np.float32),
        "fold_auc": np.asarray([curve["auc"] for curve in curves], dtype=np.float32),
    }
    pr_data = {
        "mean_recall": recall_grid.astype(np.float32),
        "mean_precision": mean_precision.astype(np.float32),
        "fold_aupr": np.asarray([curve["aupr"] for curve in curves], dtype=np.float32),
    }
    for curve in curves:
        fold = int(curve["fold"])
        roc_data[f"fold_{fold}_fpr"] = np.asarray(curve["fpr"], dtype=np.float32)
        roc_data[f"fold_{fold}_tpr"] = np.asarray(curve["tpr"], dtype=np.float32)
        roc_data[f"fold_{fold}_thresholds"] = np.asarray(curve["roc_thresholds"], dtype=np.float32)
        pr_data[f"fold_{fold}_precision"] = np.asarray(curve["precision"], dtype=np.float32)
        pr_data[f"fold_{fold}_recall"] = np.asarray(curve["recall"], dtype=np.float32)
        pr_data[f"fold_{fold}_thresholds"] = np.asarray(curve["pr_thresholds"], dtype=np.float32)
    np.savez_compressed(output_dir / "five_fold_roc_curve_data.npz", **roc_data)
    np.savez_compressed(output_dir / "five_fold_pr_curve_data.npz", **pr_data)


def save_five_fold_curves(output_dir: Path, curves):
    fpr_grid, mean_tpr, recall_grid, mean_precision = build_mean_curve_data(curves)
    mean_auc = float(np.mean([curve["auc"] for curve in curves]))
    mean_aupr = float(np.mean([curve["aupr"] for curve in curves]))

    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    for curve in curves:
        fold = int(curve["fold"])
        axis.plot(
            curve["fpr"],
            curve["tpr"],
            linewidth=1.2,
            alpha=0.75,
            label=f"ROC fold {fold} (AUC = {curve['auc']:.4f})",
        )
    axis.plot(
        fpr_grid,
        mean_tpr,
        linewidth=2.0,
        color="black",
        label=f"Mean ROC (AUC = {mean_auc:.4f})",
    )
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(f"{CONFIG.folds}-fold ROC curve")
    axis.set_xlim(-0.02, 1.05)
    axis.set_ylim(-0.05, 1.05)
    axis.legend(loc="lower right", frameon=True)
    figure.tight_layout()
    figure.savefig(
        output_dir / "five_fold_roc_curve.pdf",
        format="pdf",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    for curve in curves:
        fold = int(curve["fold"])
        axis.plot(
            curve["recall"],
            curve["precision"],
            linewidth=1.2,
            alpha=0.75,
            label=f"PR fold {fold} (AUPR = {curve['aupr']:.4f})",
        )
    axis.plot(
        recall_grid,
        mean_precision,
        linewidth=2.0,
        color="black",
        label=f"Mean PR (AUPR = {mean_aupr:.4f})",
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(f"{CONFIG.folds}-fold PR curve")
    axis.set_xlim(0.0, 1.02)
    axis.set_ylim(0.5, 1.05)
    axis.set_yticks(np.arange(0.5, 1.01, 0.1))
    axis.grid(True, alpha=0.5)
    axis.legend(loc="lower left", frameon=True)
    figure.tight_layout()
    figure.savefig(
        output_dir / "five_fold_pr_curve.pdf",
        format="pdf",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(figure)


def print_fold_metrics(fold: int, metrics: Mapping[str, float]) -> None:
    values = "  ".join(f"{metric}={metrics[metric]:.6f}" for metric in METRIC_NAMES)
    progress(f"Fold {fold} main-model metrics: {values}")


def print_summary_metrics(summary: Mapping[str, float]) -> None:
    values = "  ".join(
        f"{metric}={summary[f'{metric}_mean']:.6f}±{summary[f'{metric}_std']:.6f}"
        for metric in METRIC_NAMES
    )
    progress(f"5-fold main-model mean±std: {values}")


def run_five_fold_experiment():
    experiment_started = time.time()
    progress("=" * 88)
    progress("Strict three-branch five-fold experiment started")
    progress(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        progress(f"CUDA device: {torch.cuda.get_device_name(DEVICE)}")
    progress(f"Random seed: {CONFIG.seed}; folds: {CONFIG.folds}")
    progress(f"Output directory: {SERVER_ROOT / OUTPUT_DIRECTORY_NAME}")
    progress("Loading external caches and similarity matrices")
    set_seed(CONFIG.seed)
    data_started = time.time()
    data = load_model_data()
    progress(
        f"Data loaded in {format_duration(time.time() - data_started)}: "
        f"miRNAs={data.association.shape[0]}, diseases={data.association.shape[1]}, "
        f"known associations={int(data.association.sum())}"
    )
    if PREPARE_ONLY:
        progress("PREPARE_ONLY=True; input validation completed, training skipped")
        return {}
    output_dir = SERVER_ROOT / OUTPUT_DIRECTORY_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    progress("Sampling fixed 1:1 positive/negative benchmark pairs")
    all_m, all_d, all_y = sample_random_balanced_pairs(data.association, CONFIG.seed)
    progress(
        f"Benchmark pairs ready: total={len(all_y)}, positives={int(all_y.sum())}, "
        f"negatives={int((all_y == 0).sum())}"
    )
    progress("Constructing balanced five-fold splits")
    folds = make_balanced_folds(all_y, CONFIG.folds, CONFIG.seed)
    fold_rows = []
    curves = []

    for fold, (train_index, test_index) in enumerate(folds, start=1):
        fold_started = time.time()
        progress("-" * 88)
        progress(f"Fold {fold}/{CONFIG.folds} started")
        train_arrays = (all_m[train_index], all_d[train_index], all_y[train_index])
        test_arrays = (all_m[test_index], all_d[test_index], all_y[test_index])
        progress(
            f"Fold {fold}: outer train={len(train_index)} "
            f"(pos={int(train_arrays[2].sum())}, neg={int((train_arrays[2] == 0).sum())}), "
            f"test={len(test_index)} "
            f"(pos={int(test_arrays[2].sum())}, neg={int((test_arrays[2] == 0).sum())})"
        )
        inner_index, validation_index = make_inner_split(
            train_arrays[2], CONFIG.seed + fold * 1009,
        )
        inner_arrays = tuple(value[inner_index] for value in train_arrays)
        validation_arrays = tuple(value[validation_index] for value in train_arrays)
        progress(
            f"Fold {fold}: inner train={len(inner_index)}, validation={len(validation_index)}"
        )

        progress(f"Fold {fold} step 1/8: selecting Biological branch epoch on inner validation")
        bio_epoch, bio_inner_auc, _, validation_bio = select_bio_epoch(
            data, inner_arrays, validation_arrays, CONFIG.seed + fold * 4001 + 11,
        )
        progress(
            f"Fold {fold} step 1/8 completed: biological_epoch={bio_epoch}, "
            f"inner_val_auc={bio_inner_auc:.6f}"
        )

        progress(f"Fold {fold} step 2/8: fitting Structural branch on inner training split")
        inner_stage = build_stage_data(
            data.association.shape, inner_arrays, CONFIG.seed + fold * 4507 + 13,
        )
        inner_expert = fit_structural_expert(
            inner_stage, inner_arrays, data, CONFIG.seed + fold * 4507 + 19,
        )
        progress(f"Fold {fold}: predicting Structural branch on inner validation")
        validation_structure = predict_structural_expert(
            inner_expert, inner_stage, validation_arrays, data,
        )
        structure_inner_auc = safe_auc(validation_arrays[2], validation_structure)
        progress(
            f"Fold {fold} step 2/8 completed: structural inner_val_auc={structure_inner_auc:.6f}"
        )

        progress(f"Fold {fold} step 3/8: selecting Completion branch epoch on inner validation")
        completion_epoch, completion_inner_auc, _, validation_completion = select_completion_epoch(
            data, inner_arrays, validation_arrays, CONFIG.seed + fold * 4703 + 31,
        )
        progress(
            f"Fold {fold} step 3/8 completed: completion_epoch={completion_epoch}, "
            f"inner_val_auc={completion_inner_auc:.6f}"
        )

        progress(f"Fold {fold} step 4/8: selecting three-branch fusion weights")
        fusion_weights, fusion_inner_auc, _ = select_fusion_weights(
            validation_arrays[2],
            validation_bio,
            validation_structure,
            validation_completion,
        )
        progress(
            f"Fold {fold} step 4/8 completed: weights="
            f"A={fusion_weights[0]:.2f}, B={fusion_weights[1]:.2f}, C={fusion_weights[2]:.2f}, "
            f"inner_val_auc={fusion_inner_auc:.6f}"
        )
        del inner_stage, inner_expert

        progress(
            f"Fold {fold} step 5/8: refitting Biological branch on full outer train for {bio_epoch} epoch(s)"
        )
        bio_model, bio_ema, _ = fit_bio_from_scratch(
            data, train_arrays, bio_epoch, CONFIG.seed + fold * 5003 + 17,
        )
        progress(f"Fold {fold}: Biological branch test prediction")
        bio_probability = predict_bio(bio_model, test_arrays, bio_ema)
        progress(f"Fold {fold} step 5/8 completed")

        progress(f"Fold {fold} step 6/8: refitting Structural branch on full outer train")
        structural_stage = build_stage_data(
            data.association.shape, train_arrays, CONFIG.seed + fold * 6007 + 23,
        )
        structural_expert = fit_structural_expert(
            structural_stage, train_arrays, data, CONFIG.seed + fold * 6007 + 29,
        )
        progress(f"Fold {fold}: Structural branch test prediction")
        structure_probability = predict_structural_expert(
            structural_expert, structural_stage, test_arrays, data,
        )
        progress(f"Fold {fold} step 6/8 completed")

        progress(
            f"Fold {fold} step 7/8: refitting Completion branch on full outer train for "
            f"{completion_epoch} epoch(s)"
        )
        completion_model, _ = fit_completion_from_scratch(
            data, train_arrays, completion_epoch, CONFIG.seed + fold * 7001 + 37,
        )
        progress(f"Fold {fold}: Completion branch test prediction")
        completion_probability = predict_completion(completion_model, test_arrays)
        progress(f"Fold {fold} step 7/8 completed")

        progress(f"Fold {fold} step 8/8: fusing predictions, evaluating and saving outputs")
        final_probability = rank_fuse_three(
            bio_probability,
            structure_probability,
            completion_probability,
            fusion_weights,
        )
        labels = test_arrays[2]
        metrics = classification_metrics(labels, final_probability)
        fold_row = {"fold": fold, **metrics}
        fold_rows.append(fold_row)
        print_fold_metrics(fold, metrics)

        fpr, tpr, roc_thresholds = roc_curve(labels, final_probability)
        precision, recall, pr_thresholds = precision_recall_curve(labels, final_probability)
        curves.append({
            "fold": fold,
            "fpr": fpr,
            "tpr": tpr,
            "roc_thresholds": roc_thresholds,
            "precision": precision,
            "recall": recall,
            "pr_thresholds": pr_thresholds,
            "auc": metrics["AUC"],
            "aupr": metrics["AUPR"],
        })

        np.savez_compressed(
            output_dir / f"fold_{fold}_predictions.npz",
            labels=labels,
            mirna_indices=test_arrays[0],
            disease_indices=test_arrays[1],
            final_probability=final_probability,
            fusion_weights=fusion_weights,
        )
        torch.save({
            "biological_branch_raw_trainable_state": {
                name: value.detach().cpu() for name, value in bio_model.named_parameters()
            },
            "biological_branch_ema_state": bio_ema.state_dict(),
            "selected_epoch": bio_epoch,
            "fusion_weights": fusion_weights.tolist(),
            "config": asdict(CONFIG),
        }, output_dir / f"fold_{fold}_biological_branch_state.pt")
        with (output_dir / f"fold_{fold}_structural_expert.pkl").open("wb") as handle:
            pickle.dump(structural_expert, handle)
        torch.save({
            "completion_branch_state_dict": {
                name: value.detach().cpu() for name, value in completion_model.state_dict().items()
            },
            "selected_epoch": completion_epoch,
            "feature_names": completion_model.feature_names,
            "full_view_id": completion_model.full_view_id,
            "fusion_weights": fusion_weights.tolist(),
            "config": asdict(CONFIG),
        }, output_dir / f"fold_{fold}_completion_branch_state.pt")

        progress(
            f"Fold {fold} step 8/8 completed; fold elapsed={format_duration(time.time() - fold_started)}"
        )
        del bio_model, bio_ema, structural_stage, structural_expert, completion_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    progress("All folds completed; calculating five-fold summary")
    summary = summarize_main_model_metrics(fold_rows)
    summary_row = {"folds": CONFIG.folds, **summary}
    progress("Saving main-model metric CSV files")
    write_csv(output_dir / "five_fold_main_model_metrics.csv", fold_rows)
    write_csv(output_dir / "five_fold_main_model_summary.csv", [summary_row])
    progress("Saving ROC/PR curve data")
    save_curve_data(output_dir, curves)
    progress("Generating 600-dpi five-fold ROC and PR PDF figures")
    save_five_fold_curves(output_dir, curves)
    print_summary_metrics(summary)
    progress(f"All outputs saved to: {output_dir}")
    progress(f"Experiment completed in {format_duration(time.time() - experiment_started)}")
    progress("=" * 88)
    return summary


def main():
    run_five_fold_experiment()


if __name__ == "__main__":
    main()
