"""heldout 子集:複合全放(最多 60)＋無版面補到 120;欄位給 mdh3(id/system/user/prompt)與評分(target/card/design)。"""
import json, random, sys
src, dst = sys.argv[1], sys.argv[2]
jobs = {j['id']: j for j in map(json.loads, open('/home/pottokao/teacher_v4/07_teacher/jobs.jsonl'))}
H = [json.loads(l) for l in open(src)]
comp = [h for h in H if h.get('design', 'none') != 'none']; plain = [h for h in H if h.get('design', 'none') == 'none']
rng = random.Random(41); rng.shuffle(comp); rng.shuffle(plain)
pick = comp[:60] + plain[:120 - min(60, len(comp))]
out = [{"id": h["id"], "card": h["card"], "design": h.get("design", "none"), "kind": h["kind"], "prompt": jobs[h["id"].split("_")[0]]["prompt"],
        "knobs": jobs[h["id"].split("_")[0]]["knobs"], "system": h["messages"][0]["content"], "user": h["messages"][1]["content"],
        "negative": "", "target": h["messages"][2]["content"]} for h in pick]
json.dump(out, open(dst, "w"), ensure_ascii=False)
print(dst, len(out), "複合", sum(o["design"] != "none" for o in out))
