import json,re,sys,collections
W=set(w.strip().lower() for f in ('/tmp/american-english','/tmp/british-english') for w in open(f,errors='ignore'))
Q=re.compile(r"「([^」]+)」|『([^』]+)』|\"([^\"]+)\"|“([^”]+)”|'([A-Z][^']{2,})'")
META=re.compile(r"\b(per (the )?rules|omitted|as instructed|the task|the user|not requested)\b",re.I)
sq=lambda t: re.sub(r"[\W_]+","",t).lower()
def stem_ok(w):
    for suf in ('','s','es','ed','ing','ly','er','est','ness','y'):
        if w.endswith(suf) and w[:len(w)-len(suf)] in W: return True
        if w[:len(w)-len(suf)]+'e' in W: return True
    return False
def run(rows,tag):
    c=collections.Counter(); ex=collections.defaultdict(list)
    for a in rows:
        t=json.loads(a['target'])['rewritten_prompt']; c['n']+=1
        pq=sq(a['prompt'])
        extra=[x for m in Q.finditer(t) for x in m.groups() if x and sq(x) and sq(x) not in pq]
        if extra: c['多印字']+=1; ex['多印字'].append((a['id'],extra[:3]))
        if META.search(t): c['後設話']+=1; ex['後設話'].append((a['id'],META.search(t).group(0)))
        bad=[w for w in re.findall(r"\b[a-z]{5,}\b",t) if w not in W and not stem_ok(w)]
        if bad: c['疑似錯字']+=1; ex['疑似錯字'].append((a['id'],bad[:4]))
    print(tag,{k:(v if k=='n' else f"{v} ({100*v/c['n']:.1f}%)") for k,v in c.items()})
    for k,v in ex.items(): print('  ',k,v[:6])
new=[json.loads(l) for l in open('/home/pottokao/teacher_v4/07_teacher/accepted.jsonl')]
run([a for a in new if a['design']!='none'],'新複合')
import random; random.seed(3)
run(random.sample([a for a in new if a['design']=='none'],2000),'無版面(抽2000)')
