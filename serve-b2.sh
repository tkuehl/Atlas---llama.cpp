#!/bin/bash
# serve-b2.sh <PTQ1_0|PQ2_0> <ctx> <famma 0|1> <logname>
M=/a/Atlas/models/bonsai2/Ternary-Bonsai-2-27B-$1.gguf
E=""; [ "$3" = 1 ] && E="ATLAS_FA_MMA_DECODE_MIN_KV=0"
exec env CUDA_VISIBLE_DEVICES=0 $E /c/tmp/prism-b2/build-cuda/bin/llama-server.exe -m $M -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -c $2 -np 1 $EXTRA --host 127.0.0.1 --port 11600 > /c/tmp/prism-b2/$4 2>&1
