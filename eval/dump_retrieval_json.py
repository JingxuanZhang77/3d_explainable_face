"""Generate retrieval-style JSON using precomputed embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def load_query_embeddings(path: Path) -> Tuple[np.ndarray, List[str]]:
    data = np.load(path, allow_pickle=True)
    if "embeddings" not in data.files:
        raise KeyError(f"{path} 缺少 'embeddings' 字段")
    embeddings = data["embeddings"].astype(np.float32)
    ids = [str(x) for x in data["ids"]]
    return embeddings, ids


def load_gallery_features(path: Path) -> Tuple[np.ndarray, List[str], List[str]]:
    data = np.load(path, allow_pickle=True)
    if "features" not in data.files:
        raise KeyError(f"{path} 缺少 'features' 字段")
    features = data["features"].astype(np.float32)
    file_paths = [str(x) for x in data["file_paths"]]
    ids = [Path(p).stem.split("_")[0] for p in file_paths]
    return features, file_paths, ids


def list_query_files(directory: Path) -> List[Path]:
    return sorted(directory.glob("*.npz"))


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-6)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-6)
    return A @ B.T


def mean_average_precision(sim: np.ndarray, query_ids: Sequence[str], gallery_ids: Sequence[str]) -> float:
    aps: List[float] = []
    for i, qid in enumerate(query_ids):
        order = np.argsort(-sim[i])
        hits = 0
        precision_sum = 0.0
        for rank, idx in enumerate(order, start=1):
            if gallery_ids[idx] == qid:
                hits += 1
                precision_sum += hits / rank
        if hits:
            aps.append(precision_sum / hits)
    return float(np.mean(aps)) if aps else 0.0


def accumulate_metrics(sim: np.ndarray, query_ids: Sequence[str], gallery_ids: Sequence[str], ks: Iterable[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    total = len(query_ids)
    for k in ks:
        hits = 0
        for i, qid in enumerate(query_ids):
            topk = np.argsort(-sim[i])[:k]
            if any(gallery_ids[idx] == qid for idx in topk):
                hits += 1
        metrics[f"rank_{k}"] = hits / total if total else 0.0
        metrics[f"top{k}_hits"] = hits
    metrics["denominator"] = total
    metrics["mAP"] = mean_average_precision(sim, query_ids, gallery_ids)
    return metrics


def filter_by_seen(query_ids: Sequence[str], gallery_id_set: set[str]) -> List[int]:
    return [i for i, qid in enumerate(query_ids) if qid in gallery_id_set]


def subset_metrics(sim: np.ndarray, query_ids: Sequence[str], gallery_ids: Sequence[str], indices: Sequence[int], ks: Iterable[int]) -> Dict[str, float]:
    if not indices:
        return {f"rank_{k}": 0.0 for k in ks} | {"mAP": 0.0, "denominator": 0, "top1_hits": 0, "top5_hits": 0, "top10_hits": 0, "top20_hits": 0}
    sub_sim = sim[indices]
    sub_ids = [query_ids[i] for i in indices]
    metrics = accumulate_metrics(sub_sim, sub_ids, gallery_ids, ks)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="导出检索结果 JSON（可用于全脸或精简特征）")
    parser.add_argument("query_embeddings", type=Path, help="query 嵌入 npz (embeddings, ids)")
    parser.add_argument("gallery_features", type=Path, help="gallery 特征 npz (features, file_paths)")
    parser.add_argument("query_dir", type=Path, help="query 对应的 .npz 数据目录")
    parser.add_argument("output", type=Path, help="输出 JSON 文件路径")
    parser.add_argument("--topk", type=int, default=5, help="存入 JSON 的 top-K 结果")
    args = parser.parse_args()

    query_embeddings, query_ids = load_query_embeddings(args.query_embeddings)
    gallery_features, gallery_files, gallery_ids = load_gallery_features(args.gallery_features)
    query_files = list_query_files(args.query_dir)

    if len(query_files) != len(query_ids):
        raise ValueError("query 文件数量与嵌入数量不匹配，请确认排序方式一致。")

    similarity = cosine_similarity_matrix(query_embeddings, gallery_features)
    gallery_id_set = set(gallery_ids)

    ks = [1, 5, 10, 20]
    metrics = accumulate_metrics(similarity, query_ids, gallery_ids, ks)
    seen_indices = filter_by_seen(query_ids, gallery_id_set)
    unseen_indices = [i for i in range(len(query_ids)) if i not in seen_indices]
    metrics["seen"] = subset_metrics(similarity, query_ids, gallery_ids, seen_indices, ks)
    metrics["unseen"] = subset_metrics(similarity, query_ids, gallery_ids, unseen_indices, ks)

    results = []
    for idx, (qid, qfile) in enumerate(zip(query_ids, query_files)):
        sims = similarity[idx]
        top_indices = np.argsort(-sims)[: args.topk]
        top_files = [gallery_files[i] for i in top_indices]
        top_ids = [gallery_ids[i] for i in top_indices]
        top_sims = [float(sims[i]) for i in top_indices]

        record = {
            "query_file": str(qfile),
            "query_id": qid,
            "query_seen": qid in gallery_id_set,
            "top_5_indices": top_indices.tolist(),
            "top_5_similarities": top_sims,
            "top_5_files": top_files,
            "top_5_ids": top_ids,
            "top1_match_by_id": top_ids[0] == qid,
            "top5_match_by_id": any(gid == qid for gid in top_ids),
        }
        results.append(record)

    payload = {
        "metrics": metrics,
        "retrieval_results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"✓ JSON 写入 {args.output}")


if __name__ == "__main__":
    main()
