"""範例照網頁:examples/web/index.html(全部)+ selected.html(v5_picks.json 精選)。
編號 = v5_zbt_all_index.json;每欄一種 DiT,缺的留「—」。"""
import glob, html, json, os
import yaml
from PIL import Image
E = os.path.expanduser("~/master_dh_sp/examples/"); O = os.path.expanduser("~/ComfyUI_bh/output/examples/"); W = E + "web/"
DITS = [("zbt", "Z-Image base+turbo", ["v5_zbt", "v5b_zbt"]), ("klein", "FLUX.2 klein 4B", ["v5_klein", "v5b_klein"]),
        ("klein9", "FLUX.2 klein 9B", ["v5_klein9"]), ("qi21", "Qwen-Image 2.1", ["v5_qi21", "v5b_qi21"]),
        ("ideo4", "Ideogram 4", ["v5_ideo4_s42", "v5_ideo4_s7"]), ("boogu", "Boogu-Image Turbo", ["v5_boogu_s42", "v5_boogu_s7"]),
        ("joy", "JoyAI-Image-Edit Plus", ["v5_joy_s42", "v5_joy_s7"])]
idx = json.load(open(E + "v5_zbt_all_index.json"))
picks = {p["no"]: p["seed"] for p in json.load(open(E + "v5_picks.json"))["picks"]}
zh = {}
for f in glob.glob(os.path.expanduser("~/master_dh_sp/dist/ComfyUI-MasterStyle-Essentials/02_cards/*.yaml")):
    c = yaml.safe_load(open(f)); zh[c["code"]] = c.get("label_zh", c["code"]).split(" (")[0]
cells = {}
for key, _, tags in DITS:
    for t in tags:
        if not os.path.exists(E + t + "/log.json"): continue
        for r in json.load(open(E + t + "/log.json")):
            b = "%s_%s" % (t, r["image"].replace(".png", ""))
            if not os.path.exists(W + "img/" + b + "_t.jpg"):
                im = Image.open(O + t + "/" + r["image"]).convert("RGB"); im.save(W + "img/" + b + ".jpg", quality=90)
                im.thumbnail((400, 400)); im.save(W + "img/" + b + "_t.jpg", quality=80)
            cells.setdefault((r["code"], r["prompt"]), {}).setdefault(key, {})[r["seed"]] = (b, r.get("final_prompt", ""))
NOTE = ("擴寫:雙頭龍 + 大師 LoRA v4.1(4B:V6 dhq 2.9G + DFlash Q6;QI2.1:8B heretic dhq Q4_K_M,無草稿頭)· "
        "klein 9B 沿用 klein 4B 的擴寫,TE Qwen3-8B fp8mixed,DiT bf16 載入轉 fp8(產線是 nvfp4)· Ideogram 4 / Boogu:雙頭龍新產品,各自 8B heretic DHQ Q4_K_M 兼擴寫+TE · JoyAI:JoyAI-Image-Und 標準 Q4_K_M 兼擴寫+TE,DiT 社群 nvfp4(aha2023)· 後三者只跑精選· 點圖看原尺寸 ·「擴寫」看送進 DiT 的 prompt")
def page(title, only, size):
    rows = []
    for it in idx:
        if only and it["no"] not in picks: continue
        c = cells.get((it["card"], it["prompt"]), {}); tds = ""
        for key, _, _ in DITS:
            figs = "".join('<figure><a href="img/%s.jpg" target="_blank"><img loading="lazy" src="img/%s_t.jpg"></a><figcaption>seed %d'
                           '<details><summary>擴寫</summary><p>%s</p></details></figcaption></figure>' % (b, b, sd, html.escape(fp))
                           for sd, (b, fp) in sorted(c.get(key, {}).items(), key=lambda x: -x[0]) if not only or sd == picks[it["no"]])
            tds += "<td>%s</td>" % (figs or "<span class=na>—</span>")
        rows.append('<tr id="n%d"><th><div class=no>#%d</div><div class=card>%s</div><code>%s</code><p class=pr>%s</p></th>%s</tr>' % (
            it["no"], it["no"], html.escape(" → ".join(zh.get(p, p) for p in it["card"].split("+"))), html.escape(it["card"]),
            html.escape(it["prompt"]), tds))
    css = ("body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#ddd}header{padding:14px 18px;position:sticky;top:0;"
           "background:#111;z-index:2;border-bottom:1px solid #333}h1{font-size:18px;margin:0 0 4px}header p{margin:0;font-size:13px;color:#999}"
           "header a{color:#8bf}table{border-collapse:collapse;width:100%%}th,td{border-bottom:1px solid #2a2a2a;vertical-align:top;padding:10px}"
           "thead th{background:#1a1a1a;font-size:14px}tbody th{width:220px;text-align:left;font-weight:normal}"
           ".no{font-size:30px;color:#e55;font-weight:bold}.card{font-size:15px;color:#fff;margin:4px 0}code{font-size:11px;color:#888}"
           ".pr{font-size:13px;color:#bbb}figure{display:inline-block;margin:0 8px 8px 0;vertical-align:top}"
           "img{max-height:%dpx;max-width:%dpx;display:block;border-radius:3px}figcaption{font-size:12px;color:#888;max-width:%dpx}"
           "details p{font-size:12px;color:#aaa;white-space:pre-wrap}.na{color:#555}") % (size, size, size)
    return ('<!doctype html><html lang=zh-Hant><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">'
            '<title>%s</title><style>%s</style><header><h1>%s(%d 題)</h1><p>%s · <a href="index.html">全部</a> · <a href="selected.html">精選</a></p></header>'
            '<table><thead><tr><th>#</th>%s</tr></thead><tbody>%s</tbody></table></html>') % (
            title, css, title, len(rows), NOTE, "".join("<th>%s</th>" % n for _, n, _ in DITS), "".join(rows))
open(W + "index.html", "w").write(page("大師精選 範例照 v5 — 全部", False, 260))
open(W + "selected.html", "w").write(page("大師精選 範例照 v5 — 精選", True, 380))
print("ok")
