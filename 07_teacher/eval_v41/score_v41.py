"""v4.1 檢查點比較:同一批 heldout,各檢查點 vs 老師答案。用法: score_v41.py <in.json> <name=res.json> ..."""
import json, re, sys
sys.path.insert(0, "/home/pottokao/teacher_v41/04_node"); import master_style as ms
sys.path.insert(0, "/home/pottokao/teacher_v41/07_teacher"); sys.argv_bak = sys.argv; sys.argv = ["x"]; import driver_v41 as dv; sys.argv = sys.argv_bak
IN = json.load(open(sys.argv[1])); RUNS = [a.split("=", 1) for a in sys.argv[2:]]
PAINT = re.compile(r"\b(painting|painted|oil|watercolou?r|pastel|woodblock|engraving|ink wash|drawing|gouache|impasto|brushstroke|brushwork|on silk|pointillis|canvas|glaze)\w*", re.I)
PHOTO = re.compile(r"\b(photograph\w*|photo|photomontage|camera|lens)\b", re.I)
Q = re.compile(r'"([^"]+)"|“([^”]+)”')
squash = lambda t: re.sub(r"[\s\"“”「」『』/|\\,.，。]+", "", t).lower()
med = lambda c: "none" if c in ("none", ms.BASE) else ms.load_card(c).get("medium", "photo")
def row(x, t, ratio):
    req = [next(y for y in g if y) for g in ms.QUOTED.findall(x["prompt"])]
    m = med(x["card"]); first = re.split(r"(?<=[.!?])\s+", t.strip())[0] if t else ""
    extra = {"".join(g).strip() for g in Q.findall(t)} - set(req)
    kr = (x.get("knobs") or {}).get("ratio", "auto")
    return {"quoted_ok": sum(squash(w) in squash(t) for w in req), "quoted_n": len(req),
            "invert": (m == "painting" and (bool(PHOTO.search(first)) or not PAINT.search(t))) or (m == "photo" and bool(PAINT.search(first)) and not PHOTO.search(first)),
            "extra>2": len(extra) > 2 if x["design"] != "none" else False,
            "ratio_ok": ratio == (kr if kr != "auto" else json.loads(x["target"])["wh_ratio"]),
            "words": len(t.split()), "age": bool(dv.AGE.search(t)), "text_bad": bool(dv.text_checks(t)), "mood": bool(dv.MOOD.search(" ".join(re.split(r"(?<=[.!?])\s+", t.strip())[:-1]))),
            "name": bool(dv.NAMES.search(t)) or "master_style" in t}
def summ(rows, n):
    s = {k: sum(r[k] for r in rows) for k in rows[0]} if rows else {}
    return (f"JSON {len(rows)}/{n} | 引號 {s.get('quoted_ok',0)}/{s.get('quoted_n',0)} | 反轉 {s.get('invert',0)} | 複合多印字>2 {s.get('extra>2',0)} | "
            f"比例對 {s.get('ratio_ok',0)} | 字數均 {s.get('words',0)/max(1,len(rows)):.0f} | 年齡 {s.get('age',0)} | 文字瑕疵 {s.get('text_bad',0)} | 點名情緒 {s.get('mood',0)} | 名字/tag {s.get('name',0)}")
T = [row(x, json.loads(x["target"])["rewritten_prompt"], json.loads(x["target"])["wh_ratio"]) for x in IN]
print(f"{'老師':14s}", summ(T, len(IN)))
for name, f in RUNS:
    R = json.load(open(f)); rows = []
    for x in IN:
        r = next(iter(R.get(x["id"], {}).values()), None)
        if not r: continue
        m = re.search(r"\{.*\}", r.get("raw", ""), re.S)
        try: d = json.loads(m.group(0)); t = d.get("rewritten_prompt", ""); ratio = str(d.get("wh_ratio", ""))
        except Exception: continue
        rows.append(row(x, t, ratio))
    print(f"{name:14s}", summ(rows, len(IN)))
