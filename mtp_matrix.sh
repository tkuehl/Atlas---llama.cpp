#!/bin/bash
# MTP A/B on the grafted lean file: off / n1 / n2 (+ batch-invariant n1), depth 0 and 32K
cd /c/Users/tk199/source/repos/Atlas/scripts
OUT=eval_results/bonsai2/spec.jsonl
run() { label=$1; extra=$2; envx=$3
  taskkill //F //IM llama-server.exe >/dev/null 2>&1
  (cd /c/Users/tk199/source/repos/Atlas && docker compose stop llama-chat >/dev/null 2>&1)
  until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0) -lt 3000 ]; do sleep 3; done
  env $envx EXTRA="$extra" bash /c/tmp/prism-b2/serve-b2.sh PTQ1_0-mtp-lean 49152 1 server-$label.log &
  until curl -s 127.0.0.1:11600/health | grep -q ok; do sleep 3; done
  echo "== $label VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0)"
  PYTHONIOENCODING=utf-8 python bonsai2-eval/spec_probe.py --url http://127.0.0.1:11600 --label $label --reps 3 --out $OUT
  PYTHONIOENCODING=utf-8 python bonsai2-eval/spec_probe.py --url http://127.0.0.1:11600 --label $label --reps 2 --depth 32768 --corpus C:/tmp/prism-b2 --out $OUT
}
run off ""
run mtp-n1 "--spec-type draft-mtp --spec-draft-n-max 1"
run mtp-n2 "--spec-type draft-mtp --spec-draft-n-max 2"
run mtp-n1-binv "--spec-type draft-mtp --spec-draft-n-max 1" GGML_CUDA_BATCH_INVARIANT=1
taskkill //F //IM llama-server.exe >/dev/null 2>&1
echo MATRIX_DONE
