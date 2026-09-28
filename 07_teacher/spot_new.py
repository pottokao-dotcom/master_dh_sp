import json,re,random,sys
sys.path.insert(0,'/home/pottokao/teacher_v4/04_node'); import master_style as ms
T='/home/pottokao/teacher_v4/07_teacher/'
PAINT=re.compile(r"\b(painting|painted|oil|watercolou?r|pastel|woodblock|engraving|ink wash|ink painting|drawing|gouache|impasto|brushstroke|brushwork|on silk|pointillis|canvas|glaze)\w*",re.I)
PHOTO=re.compile(r"\b(photograph\w*|photo|photomontage|camera|lens)\b",re.I)
med=lambda c: 'none' if c in ('none',ms.BASE) else ms.load_card(c).get('medium','photo')
def stats(rows,tag):
    from collections import Counter
    c=Counter()
    for a in rows:
        if a['design']=='none': continue
        t=json.loads(a['target'])['rewritten_prompt']; m=med(a['card'])
        c[(m,'total')]+=1
        if m=='painting' and not PAINT.search(t) and PHOTO.search(t): c[(m,'畫家→照片')]+=1
        if m=='photo' and PAINT.search(t) and not PHOTO.search(t): c[(m,'照片→繪畫')]+=1
    print(tag,dict(c))
old=[json.loads(l) for l in open(T+'accepted.before_design_fix.jsonl')]
new=[json.loads(l) for l in open(T+'accepted.jsonl')]
ids={a['id'] for a in new if a['design']!='none'}
stats([a for a in old if a['id'] in ids],'舊(同一批題目)'); stats([a for a in new if a['design']!='none'],'新')
N=[a for a in new if a['design']!='none']; random.seed(11)
want=[('painting','zimage'),('painting','qwen_image'),('painting','zimage'),('photo','zimage'),('photo','qwen_image'),('none','zimage')]
used=set()
for m,e in want:
    c=[a for a in N if med(a['card'])==m and a['engine']==e and a['id'] not in used]
    a=random.choice(c); used.add(a['id']); t=json.loads(a['target'])
    lab=ms.load_card(a['card'])['label_zh'] if m!='none' else 'PE 預設(無圖卡)'
    print(f"\n=== {a['id']} [{e}] 圖卡:{lab} ({m}) + 版面:{ms.load_card(a['design'])['label_zh']}")
    print("題目:",a['prompt']); print("task_items:",t['task_items'],"ratio",t['wh_ratio'])
    print(f"照片字 {len(PHOTO.findall(t['rewritten_prompt']))} 繪畫字 {len(PAINT.findall(t['rewritten_prompt']))} 字數 {len(t['rewritten_prompt'].split())}")
    print(t['rewritten_prompt'])
