#!/bin/bash
# fetch per-layer z files from omarchy in consumer order, PAR at a time, keeping <= MAXLOC un-consumed files locally
Z=$1; PAR=${PAR:-3}; MAXLOC=${MAXLOC:-7}
mkdir -p $Z
fetch() { local l=$(printf %02d $1)
  for try in 1 2 3; do
    perl -e 'alarm 900; exec @ARGV' rsync -a -e 'ssh -o ConnectTimeout=10 -o BatchMode=yes' omarchy:/mnt/nvx/n134/cap55/zl/z$l.f16 $Z/ && { touch $Z/z$l.f16.ok; echo "[$(date +%T)] z$l"; return; }
    sleep 5
  done; echo "[$(date +%T)] z$l FAILED"; }
for l in $(seq 0 41); do
  while [ $(ls $Z/*.f16 2>/dev/null | wc -l) -ge $MAXLOC ] || [ $(jobs -r | wc -l) -ge $PAR ]; do sleep 3; done
  fetch $l &
  sleep 1
done
wait; echo "[$(date +%T)] FETCH_DONE"
