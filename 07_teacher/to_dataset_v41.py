#!/usr/bin/env python
"""accepted.jsonl → 訓練集(舊流程的 messages 格式):dataset_v41_4b.jsonl / dataset_v4_8b.jsonl + heldout(每卡約 3%)。
學生樣本 = 正常 SP 的 system/user + 學生格式答案(單行 JSON)。"""
import json, random, sys
from collections import defaultdict
from pathlib import Path
H = Path(__file__).resolve().parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else H / "accepted.jsonl"
R = [json.loads(l) for l in open(src)]
seen, by = set(), defaultdict(list)
for r in R:
    k = (r["id"], r["engine"])
    if k in seen: continue
    seen.add(k)
    if len(json.loads(r["target"]).get("task_items", [])) > 8: continue      # 篩選規則補上前產出的少數筆
    by[r["engine"]].append({"id": f"{r['id']}_{r['engine']}", "card": r["card"], "design": r["design"], "kind": r["kind"],
                            "words": len(json.loads(r["target"])["rewritten_prompt"].split()),
                            "messages": [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]},
                                         {"role": "assistant", "content": r["target"]}]})
rng = random.Random(7)
for eng, name in (("zimage", "4b"), ("qwen_image", "8b")):
    rows = by.get(eng, []); per = defaultdict(list)
    for x in rows: per[(x["card"], x["design"] != "none")].append(x)
    train, held = [], []
    for k, xs in per.items():
        rng.shuffle(xs); n = max(1, round(len(xs) * .03)) if len(xs) >= 10 else 0
        held += xs[:n]; train += xs[n:]
    rng.shuffle(train)
    (H / f"dataset_v41_{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in train))
    (H / f"heldout_v41_{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in held))
    print(name, "train", len(train), "heldout", len(held))
