#!/bin/bash
# usage: n137_watch.sh <run dir> "<since>"   B70 48:00.0 guard every 60 s while any n137-b70* container exists; on a
# trip stop every n137-b70* container at once and mark TRIPPED. Never reboots.
R=$1; since="$2"; Q=$(dirname "$0")
while docker ps --format '{{.Names}}' | grep -q '^n137-b70'; do
  if ! bash $Q/n137_guard.sh "$since" watch >> $R/guard.txt 2>&1; then
    echo "$(date '+%F %T') TRIPPED -> stopping n137-b70 containers" >> $R/guard.txt
    touch $R/TRIPPED
    pkill -f "test_ring.py|test_client_cpp.py" 2>/dev/null
    for n in $(docker ps --format '{{.Names}}' | grep '^n137-b70'); do timeout 60 docker stop -t 5 $n || timeout 30 docker kill $n; done
    exit 2
  fi
  sleep 50
done
