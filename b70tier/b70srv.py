"""N137 B70 expert server: a persistent process on one Arc Pro B70 holding a static set of GLM-5.3-Flash experts.

It spins on the shared ring (ring.py). For a LOAD it reads the keys' raw records from the NVMe store (O_DIRECT),
copies each into VRAM staging, permutes it into exl3xpu blob order (glmtier.make_perm) and points the layer's
pointer table at the slot. For a request it H2Ds the pick rows, runs exl3xpu moe_forward (_moe_n128.so, swiglu clamp
10) with one expert per row, D2Hs the weighted rows and publishes DONE_SEQ.

Env: N137_RING (/ring/ring), N137_STORE (/g/glm53_flash_exl3_3.05bpw_experts.bin), N137_SPIN_CPU (cpu to pin),
     N137_COPY (sysptr | torch), N137_LOG (json stats path), GU/DN splits (8/4), N137_READERS (8)
"""
import concurrent.futures as cf, ctypes, json, mmap, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/opt/trellis-serve/xpu")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("EXL3_MOE_LIB", "/n128/_moe_n128.so")
from exl3xpu import moe_offload
from exl3xpu.glmtier import make_perm
import ring as RG

H, I, K, E = 4096, 2048, 3, 288
libc = ctypes.CDLL(None, use_errno=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), "b70srv:", *a, flush=True)


class Server:
    def __init__(self):
        self.r = RG.Ring(os.environ.get("N137_RING", "/ring/ring"))
        self.dev = torch.device("xpu", 0)
        self.X = moe_offload.ops()
        self.X.moe_set_splits(int(os.environ.get("GU", "8")), int(os.environ.get("DN", "4")))
        self.X.moe_set_prefill_min_m(1 << 30)
        self.blob = int(self.X.blob_bytes(H, I, K))
        store = os.environ.get("N137_STORE", "/g/glm53_flash_exl3_3.05bpw_experts.bin")
        meta = json.load(open(store.rsplit(".", 1)[0] + ".json"))
        self.rec = int(meta["record_bytes"])
        assert self.rec == self.blob, (self.rec, self.blob)
        self.L = int(meta["n_layers"])
        self.fd = os.open(store, os.O_RDONLY | os.O_DIRECT)
        self.perm = make_perm(H, I, K).to(self.dev)
        self.copy_mode = os.environ.get("N137_COPY", "sysptr")
        self.slots = None
        self.slot_of = np.full(self.L * E, -1, np.int64)
        self.ptrs = torch.zeros(self.L * E, dtype=torch.int64, device=self.dev)
        # device request buffer: ids int32[MAXP] | w f32[MAXP] | x fp16[MAXP, H] (same order as the ring's REQ/X)
        self.d_ids = torch.zeros(RG.MAXP, dtype=torch.int32, device=self.dev)
        self.d_w = torch.zeros(RG.MAXP, dtype=torch.float32, device=self.dev)
        self.d_x = torch.zeros(RG.MAXP, H, dtype=torch.float16, device=self.dev)
        self.ids_addr = self.r.base + RG.OFF_REQ + 16
        self.w_addr = self.ids_addr + RG.MAXP * 4
        self.x_addr = self.r.base + RG.OFF_X
        self.out_addr = self.r.base + RG.OFF_OUT
        self.t = dict(calls=0, seen_to_done_ns=[], kern_ns=[], np=[])
        self.logp = os.environ.get("N137_LOG", "")
        p = torch.xpu.get_device_properties(self.dev)
        log(f"device {p.name} total {p.total_memory / 2**30:.2f} GiB, blob {self.blob}, store layers {self.L}, "
            f"copy {self.copy_mode}")

    # ---------------------------------------------------------------- LOAD
    def load(self, keys):
        n = len(keys)
        t0 = time.perf_counter()
        self.slots = None
        torch.xpu.empty_cache()
        self.slots = torch.empty((n, self.blob), dtype=torch.uint8, device=self.dev)
        R = 16
        stage = torch.empty((R, self.blob), dtype=torch.uint8, device=self.dev)
        sbase = stage.data_ptr()
        nbuf = 2 * R
        host = mmap.mmap(-1, nbuf * self.rec, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        hbase = ctypes.addressof(ctypes.c_char.from_buffer(host))
        libc.madvise(ctypes.c_void_p(hbase), ctypes.c_size_t(nbuf * self.rec), 14)

        def rd(key, b):
            mv = memoryview((ctypes.c_char * self.rec).from_address(hbase + b * self.rec)).cast("B")
            got = 0
            while got < self.rec:
                k = os.preadv(self.fd, [mv[got:]], key * self.rec + got)
                if k <= 0:
                    raise IOError(f"short read key {key}")
                got += k
            return b

        pool = cf.ThreadPoolExecutor(int(os.environ.get("N137_READERS", "8")))
        self.slot_of[:] = -1
        base = self.slots.data_ptr()
        for i in range(0, n, R):
            chunk = keys[i:i + R]
            half = (i // R) % 2
            futs = [pool.submit(rd, int(k), half * R + j) for j, k in enumerate(chunk)]
            for f in futs:
                f.result()
            torch.xpu.synchronize()        # previous chunk's copies from the other half are done (ordering only)
            for j in range(len(chunk)):
                self.X.memcpy_async(sbase + j * self.rec, hbase + (half * R + j) * self.rec, self.rec)
            m = len(chunk)
            d = torch.arange(i, i + m, dtype=torch.int64, device=self.dev)
            self.slots.view(torch.int32).index_copy_(0, d, stage[:m].view(torch.int32).index_select(1, self.perm))
            for j, k in enumerate(chunk):
                self.slot_of[int(k)] = i + j
        torch.xpu.synchronize()
        pool.shutdown()
        # pointer table: every key points at slot 0 unless resident (the server never runs a non-resident id)
        pt = np.full(self.L * E, base, np.int64)
        res = self.slot_of >= 0
        pt[res] = base + self.slot_of[res] * self.blob
        self.ptrs.copy_(torch.from_numpy(pt))
        torch.xpu.synchronize()
        # spot check: slot bytes == permuted record (first, middle, last)
        bad = 0
        for j in sorted({0, n // 2, n - 1}):
            raw = torch.frombuffer(bytearray(os.pread(os.open(os.environ.get("N137_STORE", "/g/glm53_flash_exl3_3.05bpw_experts.bin"), os.O_RDONLY), self.rec, int(keys[j]) * self.rec)), dtype=torch.uint8)
            want = raw.view(torch.int32)[self.perm.cpu()].contiguous()
            got = self.slots[j].view(torch.int32).cpu()
            bad += int(not torch.equal(want, got))
        host.close()
        dt = time.perf_counter() - t0
        log(f"LOAD {n} experts ({n * self.blob / 1e9:.2f} GB) in {dt:.1f} s ({n * self.rec / dt / 1e9:.2f} GB/s), "
            f"spot check bad {bad}, mem allocated {torch.xpu.memory_allocated(self.dev) / 2**30:.2f} GiB")
        return bad

    # ---------------------------------------------------------------- one request
    def serve(self, s):
        r = self.r
        t0 = time.perf_counter_ns()
        n, ntok, li = int(r.req[0]), int(r.req[1]), int(r.req[2])
        if not (0 < n <= RG.MAXP) or not (0 <= li < self.L):
            r.hdr[RG.W_ERR] = 10
            return
        ids = r.ids[:n]
        if (ids < 0).any() or (ids >= E).any() or (self.slot_of[li * E + ids.astype(np.int64)] < 0).any():
            r.hdr[RG.W_ERR] = 11           # non-resident id: refuse (the kernel would read a wrong expert)
            return
        if self.copy_mode == "sysptr":
            self.X.memcpy_async(self.d_ids.data_ptr(), self.ids_addr, n * 4)
            self.X.memcpy_async(self.d_w.data_ptr(), self.w_addr, n * 4)
            self.X.memcpy_async(self.d_x.data_ptr(), self.x_addr, n * H * 2)
        else:
            self.d_ids[:n].copy_(torch.from_numpy(ids.copy()))
            self.d_w[:n].copy_(torch.from_numpy(r.w[:n].copy()))
            self.d_x[:n].copy_(torch.from_numpy(r.x[:n]))
        out = self.X.moe_forward(self.d_x[:n], self.d_ids[:n].view(n, 1), self.d_w[:n].view(n, 1),
                                 self.ptrs[li * E:(li + 1) * E], I, K, E)
        if self.copy_mode == "sysptr":
            self.X.memcpy_async(self.out_addr, out.data_ptr(), n * H * 2)
            torch.xpu.synchronize()
        else:
            o = out.cpu()
            r.out[:n] = o.numpy()
        t1 = time.perf_counter_ns()
        self.t["calls"] += 1
        if len(self.t["np"]) < 200000:
            self.t["seen_to_done_ns"].append(t1 - t0)
            self.t["np"].append(n)

    def dump(self):
        if not self.logp or not self.t["np"]:
            return
        a, nn = np.asarray(self.t["seen_to_done_ns"]) / 1e3, np.asarray(self.t["np"])
        out = {"calls": self.t["calls"], "by_np": {}}
        for v in sorted(set(nn.tolist())):
            x = a[nn == v]
            out["by_np"][int(v)] = {"n": int(len(x)), "us_p50": round(float(np.percentile(x, 50)), 1),
                                    "us_p90": round(float(np.percentile(x, 90)), 1)}
        json.dump(out, open(self.logp, "w"), indent=1)

    def loop(self):
        r = self.r
        r.hdr[RG.W_PID] = os.getpid()
        r.hdr[RG.W_STATE] = RG.ST_NONE
        last_req, last_load = int(r.hdr[RG.W_REQ_SEQ]), int(r.hdr[RG.W_LOAD_SEQ])
        r.hdr[RG.W_DONE_SEQ] = last_req
        hb, idle = time.time(), 0
        log("ready for LOAD / requests")
        while True:
            s = int(r.hdr[RG.W_REQ_SEQ])
            if s != last_req:
                if r.hdr[RG.W_STATE] == RG.ST_READY:
                    self.serve(s)
                else:
                    r.hdr[RG.W_ERR] = 12
                last_req = s
                r.hdr[RG.W_DONE_SEQ] = s
                idle = 0
                continue
            ls = int(r.hdr[RG.W_LOAD_SEQ])
            if ls != last_load:
                r.hdr[RG.W_STATE] = RG.ST_LOADING
                n = int(r.hdr[RG.W_NKEYS])
                try:
                    bad = self.load(np.array(r.keys[:n], dtype=np.int64))
                    r.hdr[RG.W_NSLOTS] = n
                    r.hdr[RG.W_STATE] = RG.ST_READY if bad == 0 else RG.ST_ERROR
                    if bad:
                        r.hdr[RG.W_ERR] = 20
                except Exception as ex:      # noqa: BLE001
                    log("LOAD failed:", repr(ex))
                    r.hdr[RG.W_ERR] = 21
                    r.hdr[RG.W_STATE] = RG.ST_ERROR
                last_load = ls
                r.hdr[RG.W_LOAD_ACK] = ls
                continue
            if r.hdr[RG.W_QUIT]:
                break
            idle += 1
            if idle > 2000000:
                time.sleep(0.0002)
            if (idle & 0xFFFF) == 0 and time.time() - hb > 1.0:
                hb = time.time()
                r.hdr[RG.W_HEARTBEAT] = int(hb)
                if self.t["calls"]:
                    self.dump()
        self.dump()
        log("quit", self.t["calls"], "calls")


if __name__ == "__main__":
    cpu = os.environ.get("N137_SPIN_CPU")
    if cpu:
        os.sched_setaffinity(0, {int(cpu)})
    Server().loop()
