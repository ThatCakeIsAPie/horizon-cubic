#!/usr/bin/env bash
# Near fill at the surface (y 80) under HORIZON_NEAR_* / HORIZON_FAR_* settings: one launch, 3 hops each.
set -u
SRC=D:/ClaudeData/horizon-cubic
R=C:/Users/Lyle/AppData/Roaming/horizon-launcher/versions/v0.2.46/.local/horizon-client/26.3
BIN=D:/ClaudeData/near-sweep-bin
OUT=D:/ClaudeData/far-research/near/sweep
mkdir -p $BIN $OUT
cp D:/ClaudeData/horizon-cubic-target/fast/horizon-client.exe D:/ClaudeData/horizon-cubic-target/fast/horizon-ctl.exe $BIN/
for cfg in "$@"; do
  for attempt in 1 2; do
    env $( [ "$cfg" = control ] || echo $cfg ) python $SRC/tools/cubic_far_bench.py --bin-dir $BIN --resources $R \
      --detail high --y 80 --hops 3 --out $OUT/run > $OUT/run.log 2>&1
    last=$(tail -1 $OUT/run.log)
    if echo "$last" | grep -q '"near"'; then break; fi
    sleep 15
  done
  echo "$last" | python -c "
import json,sys,statistics
d=json.loads(sys.stdin.read())
near=[x for x in d.get('near',[])[1:] if x]
far=[x for x in d.get('hops',[]) if x]
print('$cfg', '| near', [round(x,2) for x in near], 'median', round(statistics.median(near),2) if near else None, '| far', [round(x,1) for x in far], '| spawn near', round(d['near'][0],2) if d.get('near') and d['near'][0] else None)
" 2>/dev/null || echo "$cfg | failed"
done
