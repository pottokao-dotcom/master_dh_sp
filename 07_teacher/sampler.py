#!/usr/bin/env python
"""v4 工作清單:題目 × 圖卡 × 版面卡 × 旋鈕 × 引擎 → jobs.jsonl(規劃見 TEACHER_V4_PLAN.md)。固定亂數種子,可重現。"""
import json, random, re, sys
from collections import defaultdict
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent / "04_node")); import master_style as ms
R = random.Random(20260927)
L = lambda f: [json.loads(l) for l in open(H / f)]
v3, newc, comp, zh, des = L("seeds_v3_clean.jsonl"), L("seeds_newcards.jsonl"), L("seeds_composite.jsonl"), L("seeds_zh.jsonl"), L("seeds_design_clean.jsonl")
ZH = {s["src_en"]: s["seed_core"] for s in zh}
normal = [s for s in v3 + newc if s["src"] != "v3_erotic"]
erotic = [s for s in v3 if s["src"] == "v3_erotic"]
own = defaultdict(list)
for s in normal:
    own[s["card"]].append(s)
PEOPLE = re.compile(r"\b(man|woman|person|people|child|boy|girl|men|women|kid|figure|couple|crowd|worker)\b|人|男|女|孩", re.I)
KID = re.compile(r"\b(child|children|boy|girl|kid|teen\w*)\b|孩|童", re.I)
COLOUR = re.compile(r"\b(red|blue|green|yellow|orange|purple|pink)\b|[紅藍綠黃紫粉]", re.I)
kids = [s for s in normal if KID.search(s["seed_core"])]
colours = [s for s in normal if COLOUR.search(s["seed_core"])]
cards = {c: ms.load_card(c) for c in ms.list_cards()}
IMG = [c for c, d in cards.items() if d.get("medium", "photo") != "design"]
DES = [c for c, d in cards.items() if d.get("medium") == "design"]
RATIOS = ["1:1", "3:2", "2:3", "16:9", "9:16", "4:3", "3:4"]

def text(s):
    t = s["seed_core"]
    return ZH.get(t, t) if t in ZH and R.random() < 0.8 else t      # 有中文翻譯的 8 成用中文

def knobs(code):
    c = cards[code]; k = {}
    x = R.random(); k["mood"] = "auto" if x < .7 else "weak" if x < .8 else "strong" if x < .9 else "off"
    x = R.random(); k["framing"] = "auto" if x < .8 else "close" if x < .9 else "wide"
    k["habitual"] = R.random() < .15
    if c.get("presets") and R.random() < .10:
        k["preset"] = R.choice(sorted(c["presets"]))
    x = R.random(); k["strength"] = "standard" if x < .8 else "light" if x < .9 else "full"
    if R.random() < .15:
        k["ratio"] = R.choice(RATIOS)
    k["seed"] = R.randrange(1 << 30)
    return k

jobs = []
for code in IMG:                                                     # 每張圖卡 100 筆
    c = cards[code]
    pool_own = own.get(code) or []
    for i in range(100):
        x = R.random()
        if c.get("adult_mood") and x < .15 and code in {s["card"] for s in erotic}:
            s, kind = R.choice([e for e in erotic if e["card"] == code]), "erotic"
        elif c.get("adult_mood") and x < .20:
            s, kind = R.choice(kids), "kid_safety"
        elif c.get("palette") == "monochrome" and x < .10:
            s, kind = R.choice(colours), "colour_named"
        elif pool_own and x < .60:
            s, kind = R.choice(pool_own), "on_theme"
        else:
            s, kind = R.choice(normal), "unrelated"
        p = text(s)
        kind += "" if PEOPLE.search(p) else "+empty"
        jobs.append({"id": f"j{len(jobs):05d}", "prompt": p, "card": code, "design": "none", "kind": kind,
                     "knobs": knobs(code), "engines": ["zimage", "qwen_image"]})
for i in range(1500):                                                # 複合
    code = R.choice([c for c in IMG if c != ms.BASE])
    s = R.choice(comp)
    jobs.append({"id": f"j{len(jobs):05d}", "prompt": s["seed_core"], "card": code, "design": R.choice(DES),
                 "kind": "composite", "knobs": knobs(code), "engines": ["qwen_image"]})
for i in range(500):                                                 # 純版面
    s = R.choice(des)
    jobs.append({"id": f"j{len(jobs):05d}", "prompt": s["seed_core"], "card": ms.BASE, "design": R.choice(DES),
                 "kind": "design_only", "knobs": knobs(ms.BASE), "engines": ["qwen_image"]})
(H / "jobs.jsonl").write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs))
from collections import Counter
print(len(jobs), "jobs,", sum(len(j["engines"]) for j in jobs), "generations")
print(Counter(j["kind"].split("+")[0] for j in jobs)); print("empty-scene", sum("+empty" in j["kind"] for j in jobs))
print("zh", sum(bool(re.search(r"[一-鿿]", j["prompt"])) for j in jobs) / len(jobs))
print("quoted", sum(bool(re.search(r'["「“]', j["prompt"])) for j in jobs) / len(jobs))
