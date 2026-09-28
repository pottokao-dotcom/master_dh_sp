#!/usr/bin/env python
"""4B/8B LoRA 驗收評分:heldout 上「新 LoRA」vs「無 LoRA 基準」。用法:python3 score_eval.py <in.json> <lora res.json> <base res.json>"""
import json, re, sys
from collections import defaultdict
sys.path.insert(0, "/home/pottokao/teacher_v4/07_teacher") if len(sys.argv) > 4 else None
IN = {x["id"]: x for x in json.load(open(sys.argv[1]))}
RES = {"lora": json.load(open(sys.argv[2])), "base": json.load(open(sys.argv[3]))}
QUOTED = re.compile(r"「([^」]+)」|『([^』]+)』|\"([^\"]+)\"|“([^”]+)”")
AGE = re.compile(r"\b(early|mid|late)[- ]?(teens|twenties|thirties|forties|fifties|sixties|seventies)\b|\b\d{1,2}[- ]?years?[- ]old", re.I)
LAMP = re.compile(r"\b(lamp|bulb|lantern|flash unit|strobe|spotlight|light fixture)s?\b", re.I)
MOOD = re.compile(r"\b(the )?(overall )?mood (is|of)\b", re.I)
MEDIUM = re.compile(r"\b(painting|oil|watercolou?r|pastel|woodblock|engraving|ink|drawing|gouache|impasto|brushstroke|silk)\b", re.I)
squash = lambda t: re.sub(r"[\s\"“”「」『』/|\\,.，。]+", "", t).lower()
def parse(raw):
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        d = json.loads(m.group(0)); return d, True
    except Exception:
        return {}, False
def user_task(u):
    m = re.search(r"Task: (.*)", u); return m.group(1) if m else ""
stats = {k: defaultdict(int) for k in RES}
for k, R in RES.items():
    for i, x in IN.items():
        r = R.get(i, {}); r = next(iter(r.values()), {}) if r else {}
        d, ok = parse(r.get("raw", ""))
        s = stats[k]; s["n"] += 1; s["json_ok"] += ok
        if not ok: continue
        t = d.get("rewritten_prompt", "") or d.get("rewrited_prompt", "")
        task = user_task(x["user"])
        s["words"] += len(t.split())
        s["has_task_items"] += "task_items" in d
        q = [next(y for y in g if y) for g in QUOTED.findall(task)]
        s["quoted_total"] += len(q); s["quoted_ok"] += sum(squash(w) in squash(t) for w in q)
        s["age"] += bool(AGE.search(t)); s["lamp"] += bool(LAMP.search(t) and not LAMP.search(task))
        s["mood_body"] += bool(MOOD.search(" ".join(re.split(r"(?<=[.!?])\s+", t.strip())[:-1])))
        s["tag_leak"] += "master_style" in t
        if "Your paintings look like this" in x["system"]:
            s["paint_n"] += 1; s["paint_medium"] += bool(MEDIUM.search(" ".join(re.split(r"(?<=[.!?])\s+", t)[:2])))
        if x.get("design", "none") != "none":
            s["design_n"] += 1; s["design_layout"] += bool(re.search(r"\b(grid|layout|margin|typograph|lettering|title|poster|cover)\b", t, re.I))
def pct(a, b): return f"{a}/{b} ({a/b:.0%})" if b else "-"
for k, s in stats.items():
    n = s["json_ok"]
    print(f"== {k}: JSON ok {pct(s['json_ok'], s['n'])} | task_items {pct(s['has_task_items'], n)} | avg words {s['words']//max(1,n)} | quoted {pct(s['quoted_ok'], s['quoted_total'])}")
    print(f"   mood-in-body {pct(s['mood_body'], n)} | numeric age {pct(s['age'], n)} | lamp {pct(s['lamp'], n)} | tag leak {s['tag_leak']} | painting names medium in opening {pct(s['paint_medium'], s['paint_n'])} | design writes layout {pct(s['design_layout'], s['design_n'])}")
