"""GPU busy fraction from an nsys sqlite export made with --cuda-graph-trace=graph.
Union of individually-launched kernels, whole-graph executions and memcpys over request windows."""
import sqlite3, sys, statistics as st
db=sqlite3.connect(sys.argv[1])
iv=[(s,e) for s,e in db.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL")]
iv+=[(s,e) for s,e in db.execute("select start,end from CUPTI_ACTIVITY_KIND_GRAPH_TRACE")]
iv+=[(s,e) for s,e in db.execute("select start,end from CUPTI_ACTIVITY_KIND_MEMCPY")]
iv.sort()
segs=[];cur=[iv[0]]
for x in iv[1:]:
    if x[0]-max(c[1] for c in cur[-50:]) > 30e6: segs.append(cur); cur=[x]
    else: cur.append(x)
segs.append(cur)
req=[s for s in segs if (s[-1][1]-s[0][0])>500e6]
span=busy=0; gaps=[]
for w in req:
    end=w[0][0]; span+=max(e for _,e in w)-w[0][0]
    for s,e in w:
        if s>end: gaps.append(s-end)
        busy+=max(0,e-max(s,end)); end=max(end,e)
print(f"windows {len(req)} span {span/1e9:.2f}s GPU busy {busy/span*100:.1f}%  idle {100-busy/span*100:.1f}%")
big=[g for g in gaps if g>20e3]
print(f"idle gaps >20us: n={len(big)} median {st.median(big)/1e3 if big else 0:.0f}us total {sum(big)/span*100:.1f}% of span; gaps <=20us total {sum(g for g in gaps if g<=20e3)/span*100:.1f}%")
