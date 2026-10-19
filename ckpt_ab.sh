#!/bin/bash
# ckpt_ab.sh <label> <env> : 8 same-class requests, report prompt ms and checkpoint copy times
label=$1; envx=$2
taskkill //F //IM llama-server.exe >/dev/null 2>&1
(cd /c/Users/tk199/source/repos/Atlas && docker compose stop llama-chat >/dev/null 2>&1)
until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0) -lt 3000 ]; do sleep 3; done
(env $envx EXTRA="--spec-type draft-mtp --spec-draft-n-max 3 -lv 5" bash /c/tmp/prism-b2/serve-b2.sh PTQ1_0-mtp-lean 49152 1 server-$label.log &)
until curl -s 127.0.0.1:11600/health | grep -q ok; do sleep 3; done
cd /c/Users/tk199/source/repos/Atlas/scripts && PYTHONIOENCODING=utf-8 python - <<'PY'
import sys, statistics; sys.path.insert(0,'bonsai2-eval')
from cache_probe import paras, chat
P=paras('C:/tmp/prism-b2'); url='http://127.0.0.1:11600'
static="You are the RESPOND writer.\n"+"\n".join(P[500:560])
ms=[]
for i in range(8):
    _,t=chat(url,[{"role":"system","content":static},{"role":"user","content":f"[call {i}] request number {i*7}"}])
    ms.append(t['prompt_ms']); 
print('prompt_ms per request:', [round(x) for x in ms], ' median(after first):', round(statistics.median(ms[1:])))
PY
python - "$label" <<'PY'
import re,sys
log=open(f'C:/tmp/prism-b2/server-{sys.argv[1]}.log',encoding='utf-8',errors='ignore').read().splitlines()
ts=lambda l: (lambda m: int(m[1])*60+int(m[2])+int(m[3])/1000 if m else None)(re.match(r'\s*(\d+)\.(\d+)\.(\d+)',l))
d=[]
for i,l in enumerate(log):
    if 'created context checkpoint' in l:
        # previous line is the do_checkpoint debug line just before the copy
        j=i-1
        while j>0 and 'do_checkpoint' not in log[j]: j-=1
        a,b=ts(log[j]),ts(l)
        if a is not None and b is not None: d.append((b-a)*1000)
print(f'checkpoint copies: n={len(d)} median {sorted(d)[len(d)//2]:.1f} ms' if d else 'no checkpoints parsed')
PY
