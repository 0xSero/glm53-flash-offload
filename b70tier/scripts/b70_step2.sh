#!/bin/bash
# N137 step 2 on B70 48:00.0 (run under n137_b70_lock.sh): server container n137-b70srv + client tests.
#   1 numerics: python ring client on the 48 cuda_ref experts   2 the engine's C++ client on the same cases (CPU-only
#   engine-image container)   3 latency + capacity: LOAD $NKEYS warm-ranked experts, round trips for 1-32 rows
set -u
R=$HOME/freetoken-exl3/runs/N137-b70tier; N128=$HOME/freetoken-exl3/runs/N128-glm53-b70; O=$R/out/${TAG:-s2}
mkdir -p $O /dev/shm/n137; chmod 777 /dev/shm/n137
[ "$N137_XPU_RENDER" = "$(readlink -f /dev/dri/by-path/pci-0000:48:00.0-render)" ] || { echo "refuse: node"; exit 9; }
since=$(date '+%F %T')
bash $R/scripts/n137_guard.sh "$since" pre | tee $O/guard_pre.txt; grep -q TRIPPED $O/guard_pre.txt && exit 3
rm -f /dev/shm/n137/ring
docker run -d --rm --name n137-b70srv --device "$N137_XPU_RENDER:$N137_XPU_RENDER:rwm" -e ZE_AFFINITY_MASK=0 \
  --cpuset-cpus 40-47 --memory 16g --memory-swap 16g --device-read-bps /dev/md127:6gb --ulimit memlock=-1:-1 --network none \
  -e NEOReadDebugKeys=1 -e EnableSharedSystemUsmSupport=1 -e EnableRecoverablePageFaults=1 -e EXL3_MOE_LIB=/n128/_moe_n128.so \
  -e N137_PROF=${PROF:-} ${SRV_ENV:-} -e N137_SPIN_CPU=41 -e N137_LOG=/o/srv_stats.json -e N137_COPY=${COPY:-sysptr} \
  -v /dev/shm/n137:/ring -v $R/b70:/b70:ro -v $N128/ovl/glmtier.py:/opt/trellis-serve/xpu/exl3xpu/glmtier.py:ro \
  -v $N128/kern/_moe_n128.so:/n128/_moe_n128.so:ro -v /mnt/nvx/glm53:/g:ro -v $R/out:/out:ro -v $O:/o -v $R/repo/data:/w:ro \
  --entrypoint bash 24c872759256 -c 'python3 -c "import sys; sys.path.insert(0,\"/b70\"); import ring; ring.Ring(\"/ring/ring\", create=True)" && exec python3 /b70/b70srv.py' > $O/srv.cid
nohup bash $R/scripts/n137_watch.sh $O "$since" > /dev/null 2>&1 &
for i in $(seq 1 120); do docker logs n137-b70srv 2>&1 | grep -q "ready for LOAD" && break; sleep 2; done
docker logs n137-b70srv 2>&1 | tail -5
if [ -f $R/out/cuda_ref.npz ] && [ "${SKIP_NUM:-0}" != 1 ]; then
  timeout 600 docker exec -e MODE=numerics -e OUT=/o/numerics.json n137-b70srv taskset -c 42 python3 /b70/test_ring.py 2>&1 | tail -40 | tee $O/numerics.log
  timeout 900 docker run --rm --name n137-b70cli --network none --cpuset-cpus 43 --memory 8g --memory-swap 8g \
    -v /dev/shm/n137:/ring -v $R/repo:/opt/glm53:ro -v $R/build:/build -e GLM53_NV2_BUILD=/build/nv2 -v $R/b70:/b70:ro -v $R/out:/out:ro \
    -v $O:/o --entrypoint bash 4732a063fa9e -c 'cd /o && python3 /b70/test_client_cpp.py' 2>&1 | grep -v "No CUDA runtime" | tail -30 | tee $O/client_cpp.log
fi
if [ "${SKIP_LAT:-0}" != 1 ]; then
  timeout 1200 docker exec -e MODE=latency -e CALLS=${CALLS:-3000} -e NKEYS=${NKEYS:-3000} -e OUT=/o/latency.json n137-b70srv taskset -c 42 python3 /b70/test_ring.py 2>&1 | tail -20 | tee $O/latency.log
fi
docker logs n137-b70srv > $O/srv.log 2>&1
timeout 30 docker exec n137-b70srv python3 -c "import sys; sys.path.insert(0,'/b70'); import ring; r=ring.Ring('/ring/ring'); r.hdr[ring.W_QUIT]=1"
sleep 3; timeout 60 docker stop -t 5 n137-b70srv >/dev/null 2>&1
bash $R/scripts/n137_guard.sh "$since" post | tee $O/guard_post.txt
