"""Run XAI-guided compact embedding pipeline for queries and gallery templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from data.preprocess import zero_mean_unit_sphere
from models.load_backbone import load_backbone
from retrieval.build_templates import (
    build_compact_templates,
    build_id_templates,
    load_templates,
    save_templates,
)
from viz.heatmap_export import save_colored_pointcloud
from xai.point_gradcam import PointGradCAM
from xai.selection import embed_points, renorm_unit_sphere, select_topk_points
from xai.targets import make_score_fn_cos_to_template, pick_top1_id_for_query


def _load_gallery_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    payload = {
        "features": data["features"].astype(np.float32),
        "labels": data["labels"],
        "file_paths": [str(x) for x in data["file_paths"]],
    }
    if "meta_json" in data.files:
        payload["meta_json"] = data["meta_json"]
    return payload


def _list_npz(directory: Path) -> List[Path]:
    return sorted(directory.glob("*.npz"))


def _compute_features_for_files(loaded, file_paths: List[Path]) -> np.ndarray:
    device = next(loaded.backbone.parameters()).device
    feats: List[np.ndarray] = []
    for path in file_paths:
        data = np.load(path)
        points = zero_mean_unit_sphere(data["points"].astype(np.float32))
        embedding = embed_points(loaded.backbone, points).cpu().numpy()[0]
        feats.append(embedding)
    return np.stack(feats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compact templates and query attributions.")
    parser.add_argument("--checkpoint", default="dgcnn_arcface_final.pth", help="Inference checkpoint path")
    parser.add_argument("--gallery-npz", default="gallery_features.npz", help="Precomputed gallery features")
    parser.add_argument("--gallery-dir", default="retrieval_data/gallery", help="Gallery directory with npz point clouds")
    parser.add_argument("--query-dir", default="retrieval_data/query", help="Query directory")
    parser.add_argument("--keep-ratio", type=float, default=0.2, help="Fraction of points kept for compact embeddings")
    parser.add_argument("--output-dir", default="outputs/xai", help="Output directory for artefacts")
    parser.add_argument("--device", default=None, help="Torch device override")
    parser.add_argument("--layer", default="conv5.2", help="Layer used for GradCAM")
    parser.add_argument("--export-heatmaps", action="store_true", help="Export coloured point clouds with saliency")
    parser.add_argument("--heatmap-dir", default="outputs/xai/heatmaps", help="Directory for heatmap PLY files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading backbone...")
    loaded = load_backbone(args.checkpoint, device=args.device)

    print("Loading gallery features...")
    gallery_npz_path = Path(args.gallery_npz)
    gallery = _load_gallery_npz(gallery_npz_path)

    gallery_files = [Path(args.gallery_dir) / Path(p).name for p in gallery["file_paths"]]

    features = gallery["features"]
    cached_meta = {}
    recomputed_gallery = False
    if "meta_json" in gallery:
        cached_meta = json.loads(str(gallery["meta_json"]))

    if cached_meta.get("fingerprint") != loaded.meta["fingerprint"]:
        print("  Cached gallery features missing or mismatched fingerprint; recomputing with current backbone...")
        features = _compute_features_for_files(loaded, gallery_files)
        cached_meta = {
            "fingerprint": loaded.meta["fingerprint"],
            "checkpoint": loaded.meta["checkpoint"],
        }
        np.savez(
            output_dir / "gallery_features_current.npz",
            features=features,
            file_paths=np.array([str(p) for p in gallery_files]),
            meta_json=json.dumps(cached_meta, sort_keys=True),
        )
        print("  ✓ Recomputed gallery features saved under outputs.")
        recomputed_gallery = True

    ids_full, templates_full, id2tmpl = build_id_templates(features, [str(p) for p in gallery_files])
    gallery_source = str(output_dir / "gallery_features_current.npz") if recomputed_gallery else str(gallery_npz_path)
    full_meta = {
        "type": "fullface",
        "fingerprint": loaded.meta["fingerprint"],
        "checkpoint": loaded.meta["checkpoint"],
        "source": gallery_source,
    }
    save_templates(output_dir / "templates_fullface.npz", ids_full, templates_full, full_meta)
    print(f"Saved full-face templates for {len(ids_full)} identities")

    compact_path = output_dir / f"templates_compact_p{args.keep_ratio:.2f}.npz"
    compact_gallery_path = output_dir / f"gallery_compact_features_p{args.keep_ratio:.2f}.npz"
    rebuild_compact = True
    if compact_path.exists():
        _, _, meta = load_templates(compact_path)
        if meta.get("fingerprint") == loaded.meta["fingerprint"] and meta.get("keep_ratio") == f"{args.keep_ratio:.4f}" and meta.get("layer") == args.layer:
            rebuild_compact = False
            print("Reusing cached compact templates")
        else:
            print("Cached compact templates mismatch metadata; rebuilding")

    if recomputed_gallery:
        rebuild_compact = True

    if rebuild_compact:
        gallery_files = [str(Path(args.gallery_dir) / Path(p).name) for p in gallery["file_paths"]]
        ids_compact, templates_compact, id2compact, compact_meta, compact_features = build_compact_templates(
            loaded,
            gallery_files,
            id2tmpl,
            keep_ratio=args.keep_ratio,
            layer=args.layer,
        )
        save_templates(compact_path, ids_compact, templates_compact, compact_meta)
        np.savez(
            compact_gallery_path,
            features=compact_features,
            file_paths=np.array(gallery_files),
            meta_json=json.dumps({**compact_meta, "type": "compact_gallery"}, sort_keys=True),
        )
        print(f"Saved compact templates for {len(ids_compact)} identities")
    else:
        ids_compact, templates_compact, compact_meta = load_templates(compact_path)
        if not compact_gallery_path.exists():
            print("⚠️ 未找到紧凑版 gallery 特征文件，请重新构建 compact 模板以生成。")

    grad_cam = PointGradCAM(loaded.backbone, layer=args.layer)
    query_files = _list_npz(Path(args.query_dir))
    query_ids: List[str] = []
    full_embeddings: List[np.ndarray] = []
    compact_embeddings: List[np.ndarray] = []
    top1_records: List[Dict[str, str]] = []

    if args.export_heatmaps:
        heatmap_dir = Path(args.heatmap_dir)
        heatmap_dir.mkdir(parents=True, exist_ok=True)

    for qpath in query_files:
        data = np.load(qpath)
        points = zero_mean_unit_sphere(data["points"].astype(np.float32))
        qid = Path(qpath).stem.split("_")[0]

        full_embed = embed_points(loaded.backbone, points).cpu().numpy()[0]
        query_ids.append(qid)
        full_embeddings.append(full_embed)

        top1_id, sim = pick_top1_id_for_query(full_embed, ids_full, templates_full)
        top1_records.append({
            "query_file": str(qpath),
            "query_id": qid,
            "top1_id": top1_id,
            "similarity": f"{sim:.6f}",
        })

        target_template = id2tmpl[top1_id]
        score_fn = make_score_fn_cos_to_template(target_template, device=next(loaded.backbone.parameters()).device)
        saliency = grad_cam.attribute(torch.from_numpy(points).to(next(loaded.backbone.parameters()).device), score_fn)[0]
        selected = select_topk_points(points, saliency, args.keep_ratio)
        selected = renorm_unit_sphere(selected)
        compact_embed = embed_points(loaded.backbone, selected).cpu().numpy()[0]
        compact_embeddings.append(compact_embed)

        if args.export_heatmaps:
            save_colored_pointcloud(points, saliency, Path(args.heatmap_dir) / f"{qid}_top1-{top1_id}.ply")

    np.savez(output_dir / "query_full_embeddings.npz", embeddings=np.stack(full_embeddings), ids=np.array(query_ids), meta_json=json.dumps(full_meta, sort_keys=True))
    np.savez(output_dir / f"query_compact_p{args.keep_ratio:.2f}.npz", embeddings=np.stack(compact_embeddings), ids=np.array(query_ids), meta_json=json.dumps({**compact_meta, "type": "compact"}, sort_keys=True))

    with (output_dir / "top1_records.json").open("w", encoding="utf-8") as fh:
        json.dump(top1_records, fh, indent=2, ensure_ascii=False)

    print(f"Processed {len(query_files)} queries")


if __name__ == "__main__":
    main()
