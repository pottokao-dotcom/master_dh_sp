#!/usr/bin/env python
"""v4 老師資料檢核表:accepted.jsonl 每筆逐條檢查 + 抽樣重組比對。用法:python3 audit.py accepted.jsonl [抽樣數]"""
import json, re, sys, random
from collections import Counter, defaultdict
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent / "04_node")); import master_style as ms
TAGS = {t["code"]: t["tag"] for t in json.load(open(H.parent / "01_system/style_tags.json"))}
SP = {k: p.read_text(encoding="utf-8").strip() for k, p in ms.SYSTEM.items()}
NAMES = re.compile(open(H / "names.regex", encoding="utf-8").read().strip(), re.I)
QUOTED = re.compile(r"「([^」]+)」|『([^』]+)』|\"([^\"]+)\"|“([^”]+)”")
squash = lambda t: re.sub(r"[\s\"“”「」『』/|\\,.，。]+", "", t).lower()
LEN = {"zimage": "Length: 250 to 350 words", "qwen_image": "Length: 450 to 550 words", "composite": "Length: 600 to 700 words",
       "composite_zimage": "Length: 350 to 400 words"}

def checklist(r):
    bad = []
    card, des, eng, s, u = r["card"], r["design"], r["engine"], r["system"], r["user"]
    c = ms.load_card(card); med = c.get("medium", "photo"); composite = des != "none"
    # 1 tag
    tags = re.findall(r"\((master_style\d\d)\)", s)
    want = [TAGS[card]] + ([TAGS[des]] if composite else [])
    if tags != want: bad.append(f"C1 tag {tags}≠{want}")
    # 2 SP 版本
    short = ms.SP_OF_ENGINE[eng] == "short"
    key = (("roleplay_design_short" if composite else "roleplay_multi_short") if short else "roleplay_multi") if (composite or med != "photo") \
        else ("roleplay_short" if short else "roleplay_long")
    if not s.startswith(SP[key]): bad.append(f"C2 SP≠{key}")
    # 3 卡片畫面段有注入、別張卡沒混進來
    look = ms.flat(c["look_paragraph"]) if c.get("look_paragraph") else ms.flat(c["look"])   # PE 預設卡是舊格式
    body = ms.strip_bw(look) if ms.user_colour(r["prompt"]) and c.get("palette") in ("monochrome", "either") else look
    if composite and card == ms.BASE and ms.load_card(des).get("picture_paragraph"):
        body = ms.flat(ms.load_card(des)["picture_paragraph"])      # 純版面:圖段換成版面卡的 picture_paragraph
    if body[:120] not in s: bad.append("C3 card look missing")
    if composite and ms.flat(ms.load_card(des)["look_paragraph"])[:120] not in s: bad.append("C4 design look missing")
    if not composite and "## The design" in s: bad.append("C4 stray design section")
    # 5 user
    if r["prompt"].strip() not in u: bad.append("C5 prompt not in user")
    verb = "Make exactly" if (composite or med != "photo") else "Photograph exactly"
    if verb not in u: bad.append("C5 verb")
    # 6 長度行
    if LEN[("composite_zimage" if eng == "zimage" else "composite") if composite else eng] not in s: bad.append("C6 length line")
    # 7 規則開關
    nude_asked = bool(ms.NUDE_ASKED.search(r["prompt"]))
    if ("\nClothing:" in s) == nude_asked: bad.append("C7 clothing rule")
    if ("\nAdults only:" in s) != bool(c.get("adult_mood")): bad.append("C7 adults-only rule")
    if ("Scope: the image-maker" in s) != composite: bad.append("C7 scope rule")
    if ("nude, naked, topless" in r["negative"]) == nude_asked: bad.append("C7 nude negative")
    # 8 答案
    try: t = json.loads(r["target"])
    except Exception: return bad + ["C8 target not json"]
    if set(t) != {"task_items", "rewritten_prompt", "wh_ratio"}: bad.append("C8 keys")
    txt = t["rewritten_prompt"]
    for q in QUOTED.findall(r["prompt"]):
        q = next(x for x in q if x)
        if squash(q) not in squash(txt): bad.append("C8 quoted " + q[:15])
    if len(t["task_items"]) > 8: bad.append("C8 items>8")
    if "master_style" in txt or NAMES.search(txt): bad.append("C8 name/tag leak")
    # 9 老師格式沒漏進學生
    if '"opening"' in s or '"regions"' in s: bad.append("C9 teacher format in student system")
    return bad

def rebuild(r):
    k = dict(r["knobs"]); seed = k.pop("seed", 0)
    s, u, n, a = ms.build(r["prompt"], r["card"], engine=r["engine"], fmt="roleplay2", design=r["design"], seed=seed,
                          mood=k.get("mood", "auto"), framing=k.get("framing", "auto"), strength=k.get("strength", "standard"),
                          habitual=k.get("habitual", False), preset=k.get("preset", "none"), ratio=k.get("ratio", "auto"))
    return s == r["system"] and u == r["user"] and n == r["negative"]

if __name__ == "__main__":
    R = [json.loads(l) for l in open(sys.argv[1])]
    per = defaultdict(int); ex = {}
    for r in R:
        for b in checklist(r):
            k = b.split(" ")[0] + " " + b.split(" ")[1]; per[k] += 1; ex.setdefault(k, (r["id"], r["engine"], b))
    print(f"records {len(R)}  | checklist failures:", dict(per) or "none")
    for k, v in ex.items(): print("  e.g.", v)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    random.seed(1); S = random.sample(R, min(n, len(R)))
    ok = sum(rebuild(r) for r in S)
    print(f"rebuild identical (system+user+negative): {ok}/{len(S)}")
    print("coverage: cards", len({r['card'] for r in R}), "designs", len({r['design'] for r in R} - {'none'}),
          "engines", Counter(r['engine'] for r in R), "kinds", Counter(r['kind'].split('+')[0] for r in R))
