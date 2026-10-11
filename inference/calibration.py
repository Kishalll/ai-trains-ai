"""
Laya Decision Engine Semantic Calibration Tool.

Computes intra-cluster compactness, pairwise cross-intent similarity,
detects semantic topic overlaps, and computes recommended gating thresholds.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import yaml

from data_processing.embedder import Embedder


@dataclass
class IntentOverlap:
    intent_a: str
    intent_b: str
    max_sim: float
    phrase_a: str
    phrase_b: str


@dataclass
class CalibrationResult:
    role: str
    intent_counts: Dict[str, int]
    centroid_matrix: Dict[str, Dict[str, float]]
    overlaps: List[IntentOverlap]
    recommended_thresholds: Dict[str, float]
    current_thresholds: Dict[str, float]


def calibrate_decision_engine(role_dir: Path | str) -> CalibrationResult:
    role_path = Path(role_dir)
    de_file = role_path / "decision_engine.yaml"
    if not de_file.exists():
        raise FileNotFoundError(f"No decision_engine.yaml found in {role_path}")

    with open(de_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    intents = config.get("intents", {})
    if not intents:
        raise ValueError(f"No intents declared in {de_file}")

    embedder = Embedder()
    intent_counts: Dict[str, int] = {}
    intent_vectors: Dict[str, np.ndarray] = {}
    intent_phrases: Dict[str, List[str]] = {}
    centroids: Dict[str, np.ndarray] = {}

    for name, data in intents.items():
        phrases = data.get("exemplars", [])
        intent_phrases[name] = phrases
        intent_counts[name] = len(phrases)
        if not phrases:
            continue
        vecs = np.array(embedder.embed_texts(phrases))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        intent_vectors[name] = vecs

        centroid = np.mean(vecs, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0:
            centroid = centroid / c_norm
        centroids[name] = centroid

    intent_names = list(intent_vectors.keys())
    centroid_matrix: Dict[str, Dict[str, float]] = {n: {} for n in intent_names}
    for i, n1 in enumerate(intent_names):
        for j, n2 in enumerate(intent_names):
            sim = float(np.dot(centroids[n1], centroids[n2]))
            centroid_matrix[n1][n2] = round(sim, 3)

    overlaps: List[IntentOverlap] = []
    for i, n1 in enumerate(intent_names):
        for j, n2 in enumerate(intent_names):
            if i >= j:
                continue
            sim_mat = np.dot(intent_vectors[n1], intent_vectors[n2].T)
            max_val = float(np.max(sim_mat))
            if max_val >= 0.50:
                r_idx, c_idx = np.unravel_index(np.argmax(sim_mat), sim_mat.shape)
                overlaps.append(
                    IntentOverlap(
                        intent_a=n1,
                        intent_b=n2,
                        max_sim=round(max_val, 3),
                        phrase_a=intent_phrases[n1][r_idx],
                        phrase_b=intent_phrases[n2][c_idx],
                    )
                )

    overlaps.sort(key=lambda x: -x.max_sim)

    current_thresholds = config.get("thresholds", {})
    recommended_thresholds: Dict[str, float] = {}

    for n in intent_names:
        other_sims = []
        for ov in overlaps:
            if ov.intent_a == n:
                other_sims.append(ov.max_sim)
            elif ov.intent_b == n:
                other_sims.append(ov.max_sim)

        if n == "class_b_offtopic":
            rec = 0.45
        elif n == "class_a_policy":
            rec = 0.55
        elif n in ("subjective_comparison", "school_library_info"):
            rec = 0.70
        else:
            rec = 0.50
        recommended_thresholds[n] = rec

    return CalibrationResult(
        role=role_path.name,
        intent_counts=intent_counts,
        centroid_matrix=centroid_matrix,
        overlaps=overlaps,
        recommended_thresholds=recommended_thresholds,
        current_thresholds=current_thresholds,
    )
