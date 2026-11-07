#!/usr/bin/env bash
# Near-terrain loading in the game, before tonight (743db86f57 + the status field) and now (research/far-fill).
set -u
SRC=D:/ClaudeData/horizon-cubic
R=C:/Users/Lyle/AppData/Roaming/horizon-launcher/versions/v0.2.46/.local/horizon-client/26.3
T=D:/ClaudeData/horizon-cubic-target
OUT=D:/ClaudeData/far-research/near
CARGO="cargo +stable-x86_64-pc-windows-msvc"
mkdir -p $OUT D:/ClaudeData/near-new-bin D:/ClaudeData/near-old-bin

echo "=== build new $(date +%H:%M:%S)"
(cd $SRC && $CARGO build -p horizon-client --bins --profile fast --target-dir $T 2>&1 | tail -1)
cp $T/fast/horizon-client.exe $T/fast/horizon-ctl.exe D:/ClaudeData/near-new-bin/

echo "=== build old $(date +%H:%M:%S)"
rm -rf D:/ClaudeData/hc-old
(cd $SRC && git worktree add -q --detach D:/ClaudeData/hc-old 743db86f57)
(cd D:/ClaudeData/hc-old && git apply D:/ClaudeData/far-research/status.patch && git diff --stat | tail -1)
(cd D:/ClaudeData/hc-old && $CARGO build -p horizon-client --bins --profile fast --target-dir $T 2>&1 | tail -1)
cp $T/fast/horizon-client.exe $T/fast/horizon-ctl.exe D:/ClaudeData/near-old-bin/
(cd $SRC && git worktree remove --force D:/ClaudeData/hc-old)

for run in 1 2; do
  for v in old new; do
    echo "=== $v run $run $(date +%H:%M:%S)"
    python $SRC/tools/cubic_far_bench.py --bin-dir D:/ClaudeData/near-$v-bin --resources $R --detail high \
      --y 80 --hops 3 --out $OUT/$v$run > $OUT/$v$run.log 2>&1
    grep -E "near terrain|^spawn|^hop" $OUT/$v$run.log | cut -c1-140
    tail -1 $OUT/$v$run.log | python -c "import json,sys; d=json.loads(sys.stdin.read()); print('near', [round(x,2) if x else x for x in d.get('near',[])], 'far hops', [round(x,2) if x else x for x in d.get('hops',[])], 'spawn', d.get('spawn_fill_s'))"
  done
done
echo "=== done $(date +%H:%M:%S)"
