"""範例照總表(編號版):python example_sheet.py <輸出名> <tag> [<tag>...]
每列 = 一題(卡+題目,複合卡寫成 卡+版面),左邊標 #編號;編號依 tag 順序、題目第一次出現的順序,
所以同一組 tag 順序在不同 DiT 上編號一致。每格一個 seed。"""
import json, os, sys
from PIL import Image, ImageDraw
name, tags, H = sys.argv[1], sys.argv[2:], 360
E, O = os.path.expanduser("~/master_dh_sp/examples/"), os.path.expanduser("~/ComfyUI_bh/output/examples/")
rows = {}
for t in tags:
    if os.path.exists(E + t + "/log.json"):
        for r in json.load(open(E + t + "/log.json")):
            rows.setdefault((r["code"], r["prompt"]), {})[r["seed"]] = O + t + "/" + r["image"]
keys = list(rows)
index = [{"no": n + 1, "card": c, "prompt": p} for n, (c, p) in enumerate(keys)]
json.dump(index, open(E + name + "_index.json", "w"), ensure_ascii=False, indent=1)
per = (len(keys) + 1) // 2
out = []
for part in (keys[:per], keys[per:]):
    strips = []
    for k in part:
        n = keys.index(k) + 1
        cells = [(Image.open(p).convert("RGB"), sd) for sd, p in sorted(rows[k].items(), key=lambda x: -x[0])]
        cells = [(im.resize((int(im.width * H / im.height), H)), sd) for im, sd in cells]
        s = Image.new("RGB", (300 + sum(im.width for im, _ in cells), H), "white"); d = ImageDraw.Draw(s)
        d.text((6, 6), "#%d" % n, fill="red", font_size=40); d.text((6, 56), k[0], fill="black", font_size=14)
        for j in range(0, min(len(k[1]), 200), 34):
            d.text((6, 80 + j // 34 * 18), k[1][j:j + 34], fill="gray", font_size=13)
        x = 300
        for im, sd in cells:
            s.paste(im, (x, 0)); d.text((x + 6, 6), "s%d" % sd, fill="red", font_size=18); x += im.width
        strips.append(s)
    S = Image.new("RGB", (max(s.width for s in strips), H * len(strips)), "white")
    for i, s in enumerate(strips):
        S.paste(s, (0, i * H))
    f = E + "%s_sheet%d.jpg" % (name, len(out) + 1); S.save(f, quality=82); out.append(f)
print(out, len(keys), "rows")
