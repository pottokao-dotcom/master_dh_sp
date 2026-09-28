"""README 範例照大表:v5 精選(v5_picks.json)× 各 DiT,輸出縮圖 + 一段 HTML 表格(GitHub README 可直接貼)。
左欄 = 編號 / 風格 / 題目(使用者原本的一句話),右邊每欄一種 DiT(同一題同一個 seed)。
用法:python make_showcase.py <輸出資料夾> <README 裡圖片的相對路徑前綴> [en|zh]  → 印出表格,圖寫進 <輸出資料夾>"""
import glob, html, json, os, sys
import yaml
from PIL import Image
E = os.path.expanduser("~/master_dh_sp/examples/"); O = os.path.expanduser("~/ComfyUI_bh/output/examples/")
DITS = [("zbt", "Z-Image<br>base+turbo", ["v5_zbt", "v5b_zbt"]), ("klein", "FLUX.2<br>klein 4B", ["v5_klein", "v5b_klein"]),
        ("klein9", "FLUX.2<br>klein 9B", ["v5_klein9"]), ("qi21", "Qwen-Image<br>2.1", ["v5_qi21", "v5b_qi21"]),
        ("ideo4", "Ideogram 4", ["v5_ideo4_s42", "v5_ideo4_s7"]), ("boogu", "Boogu-Image<br>Turbo", ["v5_boogu_s42", "v5_boogu_s7"]),
        ("joy", "JoyAI-Image<br>Edit Plus", ["v5_joy_s42", "v5_joy_s7"])]


def build(out_dir, prefix, lang="en", width=150):
    os.makedirs(out_dir, exist_ok=True)
    picks = json.load(open(E + "v5_picks.json"))["picks"]
    cards = {}
    for f in glob.glob(os.path.expanduser("~/master_dh_sp/dist/ComfyUI-MasterStyle-Essentials/02_cards/*.yaml")):
        c = yaml.safe_load(open(f)); cards[c["code"]] = c
    img = {}
    for key, _, tags in DITS:
        for t in tags:
            if os.path.exists(E + t + "/log.json"):
                for r in json.load(open(E + t + "/log.json")):
                    img[(key, r["code"], r["prompt"], r["seed"])] = O + t + "/" + r["image"]
    head = "<th>%s</th>" % ("#" if lang == "en" else "#") + "".join("<th>%s</th>" % n for _, n, _ in DITS)
    rows = []
    for n, p in enumerate(picks, 1):
        parts = p["card"].split("+")
        zh = " → ".join(cards[c].get("label_zh", c).split(" (")[0] for c in parts)
        label = "<b>#%d</b><br>%s<br><sub><code>%s</code></sub><br><sub>%s</sub>" % (
            n, html.escape(zh), html.escape(" + ".join(parts)), html.escape(p["prompt"]))
        tds = []
        for key, _, _ in DITS:
            src = img.get((key, p["card"], p["prompt"], p["seed"]))
            if not src:
                tds.append("<td align=center>—</td>"); continue
            name = "%02d_%s.jpg" % (n, key)
            im = Image.open(src).convert("RGB"); im.thumbnail((512, 512)); im.save(os.path.join(out_dir, name), quality=82)
            tds.append('<td><img src="%s%s" width="%d"></td>' % (prefix, name, width))
        rows.append("<tr><th align=left width=190>%s</th>%s</tr>" % (label, "".join(tds)))
    return "<table>\n<tr>%s</tr>\n%s\n</table>" % (head, "\n".join(rows))


if __name__ == "__main__":
    print(build(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "en"))
