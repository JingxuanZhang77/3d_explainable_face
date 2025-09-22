"""Evaluation helpers for baseline vs compact retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-6)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-6)
    return A @ B.T


def _mean_average_precision(sim: np.ndarray, query_ids: Sequence[str], template_ids: Sequence[str]) -> float:
    aps = []
    for i, qid in enumerate(query_ids):
        order = np.argsort(-sim[i])
        hits = 0
        precision_sum = 0.0
        for rank, idx in enumerate(order, start=1):
            if template_ids[idx] == qid:
                hits += 1
                precision_sum += hits / rank
        if hits:
            aps.append(precision_sum / hits)
    return float(np.mean(aps)) if aps else 0.0


def evaluate_setting(query_emb: np.ndarray, query_ids: Sequence[str], template_emb: np.ndarray, template_ids: Sequence[str], ks: Iterable[int] = (1, 5, 10, 20)) -> Dict[str, float]:
    sim = cosine_similarity_matrix(query_emb, template_emb)
    metrics: Dict[str, float] = {}
    for k in ks:
        correct = 0
        for i, qid in enumerate(query_ids):
            topk = np.argsort(-sim[i])[:k]
            if any(template_ids[j] == qid for j in topk):
                correct += 1
        metrics[f"rank_{k}"] = correct / len(query_ids) if query_ids else 0.0
    metrics["mAP"] = _mean_average_precision(sim, query_ids, template_ids)
    return metrics


def load_embeddings(path: Path) -> Tuple[np.ndarray, List[str]]:
    data = np.load(path, allow_pickle=True)
    if "embeddings" in data.files:
        array = data["embeddings"]
    elif "templates" in data.files:
        array = data["templates"]
    else:
        raise KeyError(f"{path} 缺少 embeddings/templates 字段")
    return array.astype(np.float32), [str(x) for x in data["ids"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs compact retrieval.")
    parser.add_argument("full_query", type=Path, help=".npz with full-face query embeddings")
    parser.add_argument("compact_query", type=Path, help=".npz with compact query embeddings")
    parser.add_argument("full_templates", type=Path, help=".npz templates for baseline")
    parser.add_argument("compact_templates", type=Path, help=".npz templates for compact pipeline")
    args = parser.parse_args()

    q_full, ids_full = load_embeddings(args.full_query)
    q_compact, ids_compact = load_embeddings(args.compact_query)
    if ids_full != ids_compact:
        raise ValueError("Query ID ordering mismatch between full and compact embeddings")

    tmpl_full, tmpl_ids_full = load_embeddings(args.full_templates)
    tmpl_compact, tmpl_ids_compact = load_embeddings(args.compact_templates)

    baseline = evaluate_setting(q_full, ids_full, tmpl_full, tmpl_ids_full)
    compact_to_full = evaluate_setting(q_compact, ids_full, tmpl_full, tmpl_ids_full)
    compact = evaluate_setting(q_compact, ids_full, tmpl_compact, tmpl_ids_compact)

    print("Baseline (full→full):", baseline)
    print("Compact→Full:", compact_to_full)
    print("Compact→Compact:", compact)


if __name__ == "__main__":
    main()
