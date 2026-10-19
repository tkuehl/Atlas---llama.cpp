#!/bin/bash
# greedy_cmp.sh <envA> <envB> : greedy 300-token outputs on 3 prompts, plain decode, compare
run() { envx=$1; out=$2
  taskkill //F //IM llama-server.exe >/dev/null 2>&1
  (cd /c/Users/tk199/source/repos/Atlas && docker compose stop llama-chat >/dev/null 2>&1)
  until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0) -lt 3000 ]; do sleep 3; done
  (env $envx EXTRA="" bash /c/tmp/prism-b2/serve-b2.sh PTQ1_0 16384 1 server-greedy.log &)
  until curl -s 127.0.0.1:11600/health | grep -q ok; do sleep 3; done
  python - "$out" <<'PY'
import json,sys,urllib.request
P=["Explain how a heat pump works in winter, step by step.","Write a Python function that merges overlapping intervals, with tests.","List five uses of the number 1729 in mathematics and culture."]
res=[]
for p in P:
    r=json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11600/v1/chat/completions",data=json.dumps({"messages":[{"role":"user","content":p}],"temperature":0,"top_k":1,"max_tokens":300,"reasoning_effort":"none"}).encode(),headers={"Content-Type":"application/json"})).read())
    res.append(r["choices"][0]["message"]["content"])
json.dump(res,open(sys.argv[1],"w"))
PY
}
run "$1" /c/tmp/prism-b2/greedyA.json; run "$2" /c/tmp/prism-b2/greedyB.json
taskkill //F //IM llama-server.exe >/dev/null 2>&1
python - <<'PY'
import json
a=json.load(open('C:/tmp/prism-b2/greedyA.json')); b=json.load(open('C:/tmp/prism-b2/greedyB.json'))
for i,(x,y) in enumerate(zip(a,b)):
    if x==y: print(f'prompt {i}: IDENTICAL ({len(x)} chars)')
    else:
        k=next(j for j in range(min(len(x),len(y))) if x[j]!=y[j]) if any(x[j]!=y[j] for j in range(min(len(x),len(y)))) else min(len(x),len(y))
        print(f'prompt {i}: differ at char {k} of {len(x)}: ...{x[max(0,k-40):k+30]!r} vs ...{y[max(0,k-40):k+30]!r}')
PY
