import json

with open("retrieval_full.json", encoding="utf-8") as f:
    full = json.load(f)["retrieval_results"]

with open("retrieval_compact.json", encoding="utf-8") as f:
    compact = json.load(f)["retrieval_results"]

mismatch = []
for full_rec, compact_rec in zip(full, compact):
    qid = full_rec["query_id"]
    full_top1 = full_rec["top_5_ids"][0]
    compact_top1 = compact_rec["top_5_ids"][0]
    if full_top1 != compact_top1:
        mismatch.append((qid, full_top1, compact_top1))

print(f"总 query 数: {len(full)}")
print(f"Top-1 一致的数量: {len(full) - len(mismatch)}")
print(f"Top-1 不一致的数量: {len(mismatch)}")
print("示例差异（最多打印前 5 条）:")
for item in mismatch[:5]:
    print(item)
