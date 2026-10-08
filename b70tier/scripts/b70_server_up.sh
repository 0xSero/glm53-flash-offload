#!/bin/bash
# N137 step 3: start the persistent B70 expert server (n137-b70srv) on 48:00.0 under the B70 lock and keep the B70
# guard running every 60 s. Run as: n137_b70_lock.sh n137-step3 scripts/b70_server_up.sh  (blocks until STOP_B70 exists)
set -u
R=$HOME/freetoken-exl3/runs/N137-b70tier; N128=$HOME/freetoken-exl3/runs/N128-glm53-b70; O=$R/runs/b70srv_$(date +%m%d_%H%M); mkdir -p $O /dev/shm/n137; chmod 777 /dev/shm/n137
[ "$N137_XPU_RENDER" = "$(readlink -f /dev/dri/by-path/pci-0000:48:00.0-render)" ] || { echo "refuse: node"; exit 9; }
since=$(date '+%F %T'); rm -f $R/STOP_B70 $R/TRIPPED
bash $R/scripts/n137_guard.sh "$since" pre | tee $O/guard_pre.txt; grep -q TRIPPED $O/guard_pre.txt && exit 3
rm -f /dev/shm/n137/ring
docker run -d --rm --name n137-b70srv --device "$N137_XPU_RENDER:$N137_XPU_RENDER:rwm" -e ZE_AFFINITY_MASK=0 \
  --cpuset-cpus 40-47 --memory 16g --memory-swap 16g --device-read-bps /dev/md127:6gb --ulimit memlock=-1:-1 --network none \
  -e NEOReadDebugKeys=1 -e EnableSharedSystemUsmSupport=1 -e EnableRecoverablePageFaults=1 -e EXL3_MOE_LIB=/n128/_moe_n128.so \
  -e N137_SPIN_CPU=41 -e N137_LOG=/o/srv_stats.json -e N137_COPY=${COPY:-sysptr} \
  -v /dev/shm/n137:/ring -v $R/b70:/b70:ro -v $N128/ovl/glmtier.py:/opt/trellis-serve/xpu/exl3xpu/glmtier.py:ro \
  -v $N128/kern/_moe_n128.so:/n128/_moe_n128.so:ro -v /mnt/nvx/glm53:/g:ro -v $O:/o \
  --entrypoint bash 24c872759256 -c 'python3 -c "import sys; sys.path.insert(0,\"/b70\"); import ring; ring.Ring(\"/ring/ring\", create=True)" && exec python3 /b70/b70srv.py' > $O/srv.cid
nohup bash $R/scripts/n137_watch.sh $O "$since" > /dev/null 2>&1 &
while docker ps --format '{{.Names}}' | grep -q '^n137-b70srv$' && [ ! -f $R/STOP_B70 ]; do sleep 10; done
docker logs n137-b70srv > $O/srv.log 2>&1
timeout 30 docker exec n137-b70srv python3 -c "import sys; sys.path.insert(0,'/b70'); import ring; r=ring.Ring('/ring/ring'); r.hdr[ring.W_QUIT]=1" 2>/dev/null
sleep 3; timeout 60 docker stop -t 5 n137-b70srv >/dev/null 2>&1
bash $R/scripts/n137_guard.sh "$since" post | tee $O/guard_post.txt
