from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import yaml

from data_processing.embedder import Embedder

class StandardIntent(str, Enum):
    GENERAL_CONVERSATION = "general_conversation"
    CORE_SEARCH = "core_search"
    CATALOG_SEARCH = "catalog_search"
    CLASS_A_POLICY = "class_a_policy"
    CLASS_B_OFFTOPIC = "class_b_offtopic"
    SUBJECTIVE_COMPARISON = "subjective_comparison"


class LayaDecisionEngine:
    """
    Generalized Non-Autoregressive System 1 Decision Engine for AI-Trains-AI roles.

    Dynamically loads intent taxonomy, exemplars, actions, and priority thresholds
    from role configuration (role_dir / 'decision_engine.yaml' or role.yaml).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        role_dir: Optional[Path | str] = None,
        entity_checker: Optional[Callable[[str], bool]] = None,
    ):
        self.role_dir = Path(role_dir) if role_dir else None
        self.config: Dict[str, Any] = dict(config or {})
        self.entity_checker = entity_checker

        # Load from decision_engine.yaml if present in role_dir
        if self.role_dir and (self.role_dir / "decision_engine.yaml").exists():
            try:
                with open(self.role_dir / "decision_engine.yaml", "r", encoding="utf-8") as f:
                    file_conf = yaml.safe_load(f) or {}
                    if isinstance(file_conf, dict):
                        self.config.update(file_conf)
            except Exception as e:
                print(f"Warning: Could not load decision_engine.yaml from {self.role_dir}: {e}")

        self.enabled: bool = bool(self.config.get("enabled", True))
        self.embedder = Embedder()
        self.intents_config: Dict[str, Dict[str, Any]] = self.config.get("intents", {})
        self.thresholds: Dict[str, float] = self.config.get("thresholds", {})
        self.priority_order: List[str] = self.config.get("priority_order", list(self.intents_config.keys()))

        self._matrices: Dict[str, np.ndarray] = {}
        self._init_matrices()

    def _init_matrices(self):
        for intent_name, intent_data in self.intents_config.items():
            exemplars = intent_data.get("exemplars", [])
            if not exemplars:
                continue
            raw_vecs = self.embedder.embed_texts(exemplars)
            norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = np.array(raw_vecs) / norms
            self._matrices[intent_name] = normalized

    def get_intent_action(self, intent_name: Optional[str]) -> Tuple[str, str]:
        """Returns (action_type, response_text_or_target) for a given intent."""
        if not intent_name or str(intent_name) == "None":
            return "generative_llm", ""
        intent_info = self.intents_config.get(intent_name, {})
        action = intent_info.get("action", "generative_llm")
        response = intent_info.get("response", "")
        return action, response

    def classify(self, text: str) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        Classifies query text against declared intent matrices.
        Returns: (winning_intent, confidence_score, all_scores_dict)
        If confidence does not meet declared threshold, returns (None, score, scores)
        to cleanly route unconfident queries to the generative LLM.
        """
        cleaned = text.strip()
        if not cleaned or not self._matrices:
            return None, 0.0, {}

        raw_q = self.embedder.embed_query(cleaned)
        q_norm = np.linalg.norm(raw_q)
        if q_norm == 0:
            q_norm = 1.0
        q_vec = np.array(raw_q) / q_norm

        scores: Dict[str, float] = {}
        for intent_name, mat in self._matrices.items():
            sims = mat.dot(q_vec)
            max_sim = float(np.max(sims))
            if len(sims) >= 3:
                top_k = np.sort(sims)[-3:]
                density = float(np.mean(top_k))
                scores[intent_name] = max_sim if max_sim >= 0.88 else float(0.5 * max_sim + 0.5 * density)
            else:
                scores[intent_name] = max_sim

        search_intent_name = "core_search" if "core_search" in scores else ("catalog_search" if "catalog_search" in scores else "")
        search_score = scores.get(search_intent_name, 0.0) if search_intent_name else 0.0

        # Check declared priority order
        for intent_name in self.priority_order:
            thresh = self.thresholds.get(intent_name, 0.50)
            score = scores.get(intent_name, 0.0)

            if intent_name in ("core_search", "catalog_search"):
                continue

            # For policy and off-topic intents, verify they beat domain search
            if intent_name in ("class_a_policy", "class_b_offtopic"):
                if score >= thresh and score > search_score:
                    return intent_name, score, scores
            elif intent_name == "general_conversation":
                # High-confidence conversational greetings, thanks, identity, or fillers immediately win
                if score >= 0.80 and search_score < 0.40:
                    return intent_name, score, scores

                # Verify query does not match a domain entity if an entity checker is provided
                if self.entity_checker and callable(self.entity_checker):
                    try:
                        if self.entity_checker(cleaned):
                            continue
                    except Exception:
                        pass
                if score >= thresh and score >= search_score:
                    return intent_name, score, scores
            else:
                # Top priority gates (subjective comparison, custom intents)
                if score >= thresh:
                    return intent_name, score, scores

        # Check domain search confidence threshold
        if search_intent_name:
            search_thresh = self.thresholds.get(search_intent_name, 0.45)
            if search_score >= search_thresh:
                return search_intent_name, search_score, scores

        # Check domain entity grounding if checker callback provided
        if self.entity_checker and callable(self.entity_checker):
            try:
                if self.entity_checker(cleaned):
                    return (search_intent_name or "core_search"), 0.95, scores
            except Exception:
                pass

        # If none of the intents met their confidence threshold, route to generative LLM
        best_score = max(scores.values()) if scores else 0.0
        return None, best_score, scores
