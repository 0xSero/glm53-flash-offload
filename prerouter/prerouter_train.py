#!/usr/bin/env python3
"""N134 phase B: train Edge0-style prerouter heads on the N134 capture and evaluate them offline.

Edge0 (arXiv 2609.18063, github.com/Edge0-AI/Edge0 python/src/edge0/prerouter): the head OWNED by layer N runs at token
t on layer N's MoE input and the one-hots of layer N's routed top-k at t and t-1, and predicts layer N+1's routing at
t+1. head = fc1 -> erf-GELU -> fc2 + linear_init (all bias-free, on the same concat features); linear_init is
warm-started from the next layer's router, fc2 starts at 0 so training begins at "apply the next router to this hidden
state". Output = router-style logits; GLM-5.3 selection = top-8 of sigmoid(logit) + e_score_correction_bias.

Variants (consumer c = MoE layer index 0..41, t = decode token):
  edge0   features of layer c-1 at t (+ one-hots c-1 at t, t-1)  -> routing of c at t+1      (paper; c=0 uses layer 41)
  self    features of layer c at t   (+ one-hots c at t, t-1)    -> routing of c at t+1      (all at the step boundary)
  same    features of layer c-1 at t (+ one-hots c-1 at t, t-1) -> routing of c at t        (same-token, Pre-gated MoE)
Baselines (no training): router_same = router c on z[t, c-1] -> (t, c)  (= today's GLM53_NV_PREFETCH),
  router_next = router c on z[t, c-1] -> (t+1, c), router_self = router c on z[t, c] -> (t+1, c) (temporal).

Loss (phase 1 distillation only): BCE between sigmoid(head) and sigmoid(true router logits of c at the target token)
+ lam * listwise softmax CE of the true top-8 under the selection score (sigmoid + bias) / tau.

Inputs: <cap>/{z,sel,w,meta,tier}.bin from nv2.Capture, router.npz, requests.jsonl (client log, for the prompt split).
Outputs in <out>: data/ (per-layer transposed z, routes, tiers, split), heads_<variant>.pt, pred_<variant>.npy
(test tokens x 42 x 32 ranked ids), metrics.json."""
import argparse, hashlib, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

E, K, NR = 288, 8, 32
MAXU_PAD = 64


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ------------------------------------------------------------------------------------------------- data preparation
def prepare(cap, out, L, H, req_log, write_z=True):
    d = os.path.join(out, "data"); os.makedirs(d, exist_ok=True)
    meta = np.fromfile(os.path.join(cap, "meta.bin"), np.int64).reshape(-1, 4)
    T = len(meta)
    sel = np.fromfile(os.path.join(cap, "sel.bin"), np.int16).reshape(-1, L, K)[:T]
    w = np.fromfile(os.path.join(cap, "w.bin"), np.float16).reshape(-1, L, K)[:T]
    assert len(sel) == T and len(w) == T, (len(sel), len(w), T)
    zf = os.path.join(cap, "z.bin")
    assert not write_z or os.path.getsize(zf) >= T * L * H * 2
    # request ids: boundary flag marks the first decode token after a prefill / small call
    req = np.cumsum(meta[:, 2]) - 1
    req[req < 0] = 0
    nreq = int(req.max()) + 1
    # tiers per pick from the host stream (li, nu, ntok, key[nu], cls[nu]) indexed by decode call
    ts = np.fromfile(os.path.join(cap, "tier.bin"), np.int32)
    RW = 3 + 2 * K
    if len(ts) % RW == 0 and (ts.reshape(-1, RW)[:, 1] == K).all():        # C1 decode: always 8 unique picks
        rr = ts.reshape(-1, RW)
        rli, rkey, rcls = rr[:, 0], rr[:, 3:3 + K], rr[:, 3 + K:]
    else:                                                                   # general stream (nu varies)
        li_, ky_, cl_, i = [], [], [], 0
        while i + 3 <= len(ts):
            nu = int(ts[i + 1])
            if i + 3 + 2 * nu > len(ts):
                break
            k_ = np.full(MAXU_PAD, -1, np.int32); c_ = np.full(MAXU_PAD, -1, np.int32)
            n_ = min(nu, MAXU_PAD)
            k_[:n_] = ts[i + 3:i + 3 + n_]; c_[:n_] = ts[i + 3 + nu:i + 3 + nu + n_]
            li_.append(ts[i]); ky_.append(k_); cl_.append(c_)
            i += 3 + 2 * nu
        rli, rkey, rcls = np.array(li_), np.stack(ky_), np.stack(cl_)
    ncall = len(rli)
    j = meta[:, 1:2] + np.arange(L)[None, :]                                # [T, L] decode-call index
    okj = j < ncall
    jj = np.where(okj, j, 0)
    lay_ok = okj & (rli[jj] == np.arange(L)[None, :])
    bad = int((~lay_ok).sum())
    tier = np.full((T, L, K), -1, np.int8)
    for a0 in range(0, T, 8192):
        sl = slice(a0, a0 + 8192)
        gkey = sel[sl].astype(np.int32) + (np.arange(L) * E)[None, :, None]  # [t, L, K] global keys
        rk, rc = rkey[jj[sl]], rcls[jj[sl]]                                 # [t, L, U]
        match = gkey[:, :, :, None] == rk[:, :, None, :]                    # [t, L, K, U]
        tier[sl] = np.where(match.any(3), (rc[:, :, None, :] * match).sum(3), -1)
    tier[~lay_ok] = -1
    recs = rli
    log(f"prepare: {T} tokens, {nreq} requests, {len(recs)} tier records, misaligned layer calls {bad}, "
        f"tier coverage {(tier >= 0).mean():.4f}")
    # request / prompt of each token from the client log by wall time (token wall ns inside [t0, t_end] of a request);
    # tokens of no logged request (e.g. the final flush request) go to a pseudo request with split 3 (unused)
    pid = [f"req{r}" for r in range(nreq)]
    if req_log and os.path.exists(req_log):
        rl = [json.loads(l) for l in open(req_log)]
        rl = [r for r in rl if r.get("t0") and r.get("t_end")]
        t0s = np.array([r["t0"] for r in rl]); tes = np.array([r["t_end"] for r in rl])
        o = np.argsort(t0s); t0s, tes, rl = t0s[o], tes[o], [rl[i] for i in o]
        wall = meta[:, 3] / 1e9
        ix = np.searchsorted(t0s, wall, "right") - 1
        ok = (ix >= 0) & (wall <= tes[np.clip(ix, 0, None)] + 1.0)
        req = np.where(ok, ix, len(rl)).astype(np.int64)
        nreq = len(rl) + 1
        pid = [r["id"] for r in rl] + ["_none"]
        ct = np.array([(r.get("usage") or {}).get("completion_tokens", 0) for r in rl])
        got = np.bincount(req, minlength=nreq)[:-1]
        log(f"time map: {len(rl)} logged requests, {int((~ok).sum())} tokens outside any request; captured/logged "
            f"tokens {got.sum()}/{ct.sum()}; boundary flags {int(meta[:, 2].sum())}; per-request |diff| p50 "
            f"{np.median(np.abs(got - ct)):.0f} max {np.abs(got - ct).max()}")
    hsh = lambda s: int(hashlib.md5(s.encode()).hexdigest()[:8], 16)
    split = np.array([3 if p == "_none" else (2 if hsh(p) % 5 == 0 else (1 if hsh(p) % 10 == 1 else 0)) for p in pid],
                     np.int8)                                               # 0 train 1 val 2 test 3 unused
    np.save(os.path.join(d, "sel.npy"), sel); np.save(os.path.join(d, "w.npy"), w)
    np.save(os.path.join(d, "tier.npy"), tier); np.save(os.path.join(d, "req.npy"), req)
    np.save(os.path.join(d, "split_req.npy"), split)
    json.dump({"T": T, "L": L, "H": H, "nreq": nreq, "pid": pid, "misaligned": bad}, open(os.path.join(d, "info.json"), "w"))
    if not write_z:
        log("prepare: per-layer z comes from --zdir")
        return
    # per-layer transposed z (fp16 [T, H]) in one sequential pass
    zm = np.memmap(zf, np.float16, "r", shape=(T, L, H))
    fs = [open(os.path.join(d, f"z{l:02d}.f16"), "wb") for l in range(L)]
    B = 2048
    for a in range(0, T, B):
        blk = np.asarray(zm[a:a + B])
        for l in range(L):
            fs[l].write(np.ascontiguousarray(blk[:, l]).tobytes())
    for f in fs:
        f.close()
    log("prepare: per-layer z written")


# ------------------------------------------------------------------------------------------------- model
class Head(torch.nn.Module):
    def __init__(self, H, hid, gate):
        super().__init__()
        fin = H + 2 * E
        self.fc1 = torch.nn.Linear(fin, hid, bias=False)
        self.fc2 = torch.nn.Linear(hid, E, bias=False)
        self.li = torch.nn.Linear(fin, E, bias=False)
        with torch.no_grad():
            torch.nn.init.zeros_(self.fc2.weight)
            self.li.weight.zero_()
            self.li.weight[:, :H] = gate          # warm start: the consumer's router on the hidden part
            self.fc1.weight.mul_(0.5)

    def forward(self, x):
        return self.li(x) + self.fc2(F.gelu(self.fc1(x)))   # F.gelu default = exact erf


def onehot(ids, valid=None):
    o = torch.zeros(ids.shape[0], E, device=ids.device)
    o.scatter_(1, ids.long().clamp(min=0), 1.0)
    if valid is not None:
        o *= valid[:, None]
    return o


class Data:
    def __init__(self, out, L, H, dev, zdir=""):
        d = os.path.join(out, "data")
        self.d, self.L, self.H, self.dev = d, L, H, dev
        self.zdir = zdir or d
        info = json.load(open(os.path.join(d, "info.json")))
        self.T = info["T"]
        self.sel = torch.from_numpy(np.load(os.path.join(d, "sel.npy")).astype(np.int64)).to(dev)
        self.tier = np.load(os.path.join(d, "tier.npy"))
        self.req = np.load(os.path.join(d, "req.npy"))
        sp = np.load(os.path.join(d, "split_req.npy"))
        self.split = sp[self.req]
        r = torch.from_numpy(self.req).to(dev)
        same_next = torch.zeros(self.T, dtype=torch.bool, device=dev); same_next[:-1] = r[1:] == r[:-1]
        same_prev = torch.zeros(self.T, dtype=torch.bool, device=dev); same_prev[1:] = r[1:] == r[:-1]
        self.has_next, self.has_prev = same_next, same_prev
        self._z = {}

    def z(self, l):
        if l not in self._z:
            if len(self._z) >= 4:
                self._z.pop(next(iter(self._z)))
            fn = os.path.join(self.zdir, f"z{l:02d}.f16")
            need = self.T * self.H * 2
            t0 = time.time()
            while not (os.path.exists(fn + ".ok") or (self.zdir == self.d and os.path.exists(fn))):
                time.sleep(5)                                   # streamed in by the fetcher (writes <file>.ok when done)
            if time.time() - t0 > 5:
                log(f"  waited {time.time() - t0:.0f} s for {fn}")
            a = np.fromfile(fn, np.float16, count=self.T * self.H).reshape(self.T, self.H)
            self._z[l] = torch.from_numpy(a).to(self.dev)
        return self._z[l]


def feats(D, l, idx):
    """features of layer l at tokens idx: [z, onehot(sel[t,l]), onehot(sel[t-1,l]) or 0]"""
    z = D.z(l)[idx].float()
    cur = onehot(D.sel[idx, l])
    pv = D.has_prev[idx].float()
    prev = onehot(D.sel[(idx - 1).clamp(min=0), l], pv)
    return torch.cat([z, cur, prev], 1)


def variant_spec(v, c, L):
    """(owner layer, token shift of the target)"""
    if v in ("edge0", "router_next"):
        return (c - 1 if c > 0 else L - 1), 1
    if v in ("self", "router_self"):
        return c, 1
    if v in ("same", "router_same"):
        return max(c - 1, 0), 0
    raise ValueError(v)


def selection(logits, bias):
    return torch.sigmoid(logits) + bias


def train_head(D, R, c, v, a, idx_tr, idx_va):
    L = D.L
    own, sh = variant_spec(v, c, L)
    gate = R["gate"][c]; bias = R["bias"][c]
    h = Head(D.H, a.hidden, gate).to(D.dev)
    opt = torch.optim.AdamW(h.parameters(), lr=a.lr, weight_decay=a.wd)
    nstep = a.epochs * math.ceil(len(idx_tr) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=nstep, pct_start=0.05)
    zc = D.z(c)

    def batch_loss(idx, train=True):
        x = feats(D, own, idx)
        tgt = idx + sh
        with torch.no_grad():
            tl = zc[tgt].float() @ gate.T                     # true router logits of c at the target token
            tsel = D.sel[tgt, c]
        lg = h(x)
        ld = F.binary_cross_entropy_with_logits(lg, torch.sigmoid(tl))
        s = selection(lg, bias) / a.tau
        lr_ = -(torch.log_softmax(s, 1).gather(1, tsel)).mean()
        return ld + a.lam * lr_, lg

    best, best_state, step = 1e9, None, 0
    g = torch.Generator(device="cpu").manual_seed(c)
    for ep in range(a.epochs):
        h.train()
        perm = idx_tr[torch.randperm(len(idx_tr), generator=g).to(D.dev)]
        for b in range(0, len(perm), a.bs):
            loss, _ = batch_loss(perm[b:b + a.bs])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
        h.eval()
        with torch.no_grad():
            vl, rc = 0.0, 0.0
            for b in range(0, len(idx_va), 8192):
                ii = idx_va[b:b + 8192]
                l_, lg = batch_loss(ii, False)
                vl += float(l_) * len(ii)
                top = selection(lg, bias).topk(K, 1).indices
                rc += float((top[:, :, None] == D.sel[ii + sh, c][:, None, :]).any(1).float().sum())
            vl /= max(1, len(idx_va)); rc /= max(1, len(idx_va) * K)
        if vl < best:
            best, best_state = vl, {k: v_.detach().clone() for k, v_ in h.state_dict().items()}
        if a.verbose:
            log(f"  {v} c{c:02d} ep{ep} val loss {vl:.4f} recall@8 {rc:.4f}")
    h.load_state_dict(best_state)
    return h, rc


@torch.no_grad()
def predict(D, R, c, v, h, idx):
    own, sh = variant_spec(v, c, D.L)
    bias = R["bias"][c]
    out = []
    for b in range(0, len(idx), 8192):
        ii = idx[b:b + 8192]
        if h is None:                                          # router baselines
            lg = D.z(own)[ii].float() @ R["gate"][c].T
        else:
            lg = h(feats(D, own, ii))
        out.append(selection(lg, bias).topk(NR, 1).indices.to(torch.int16))
    return torch.cat(out).cpu().numpy()


def metrics(pred, true, tier, Ns=(8, 12, 16, 24, 32)):
    """pred [n, L, NR] ranked, true [n, L, K], tier [n, L, K] (-1 unknown, 0 vram, 1 ram, 2 vring, 3 inflight, 4 nvme)"""
    out = {}
    hit_rank = np.full(true.shape, 10 ** 6, np.int32)          # rank of each true pick in the prediction
    for r in range(pred.shape[2]):
        m = pred[:, :, r:r + 1] == true
        hit_rank[m & (hit_rank > r)] = r
    groups = {"all": tier >= -1, "vram": tier == 0, "ram": (tier == 1) | (tier == 2), "nvme": tier >= 3}
    for N in Ns:
        hit = hit_rank < N
        for gname, gm in groups.items():
            n = gm.sum()
            out[f"recall@{N}/{gname}"] = round(float(hit[gm].sum() / max(1, n)), 4)
    out["n_picks"] = {g: int(m.sum()) for g, m in groups.items()}
    out["recall_by_layer@8"] = [round(float((hit_rank[:, l] < 8).mean()), 4) for l in range(true.shape[1])]
    nv = tier >= 3
    out["recall_by_layer@16/nvme"] = [round(float((hit_rank[:, l][nv[:, l]] < 16).mean()) if nv[:, l].any() else -1, 4)
                                      for l in range(true.shape[1])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True); ap.add_argument("--router", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--requests", default="")
    ap.add_argument("--zdir", default="", help="per-layer z files z<l>.f16 (+ .ok markers) instead of transposing cap/z.bin")
    ap.add_argument("--delete-z", action="store_true", help="delete a layer file once no remaining consumer needs it")
    ap.add_argument("--variants", default="edge0,self,same,router_same,router_next,router_self")
    ap.add_argument("--layers", default="", help="subset of consumers, e.g. 1,10,20 (screening)")
    ap.add_argument("--hidden", type=int, default=512); ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=512); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.05); ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--tag", default=""); ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--L", type=int, default=42); ap.add_argument("--H", type=int, default=4096)
    ap.add_argument("--device", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not os.path.exists(os.path.join(a.out, "data", "info.json")):
        prepare(a.cap, a.out, a.L, a.H, a.requests, write_z=not a.zdir)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
    D = Data(a.out, a.L, a.H, dev, a.zdir)
    rz = np.load(a.router)
    R = {"gate": torch.from_numpy(rz["gate"]).to(dev), "bias": torch.from_numpy(rz["bias"]).to(dev)}
    T, L = D.T, D.L
    ar = torch.arange(T, device=dev)
    sp = torch.from_numpy(D.split.astype(np.int64)).to(dev)
    cons = [int(x) for x in a.layers.split(",")] if a.layers else list(range(1, L)) + [0]
    tag = a.tag or "full"
    variants = a.variants.split(",")
    res = {"args": vars(a), "T": T, "device": str(dev), "variants": {}, "router_reproduces_capture": {}}
    V = {}
    for v in variants:
        sh = variant_spec(v, 1, L)[1]
        ok = D.has_next if sh else torch.ones(T, dtype=torch.bool, device=dev)
        idx_tr, idx_va, idx_te = ar[ok & (sp == 0)], ar[ok & (sp == 1)], ar[ok & (sp == 2)]
        if a.max_train and len(idx_tr) > a.max_train:
            idx_tr = idx_tr[:a.max_train]
        V[v] = dict(sh=sh, tr=idx_tr, va=idx_va, te=idx_te, P=np.full((len(idx_te), L, NR), -1, np.int16), heads={}, vrec={})
        log(f"variant {v}: train {len(idx_tr)} val {len(idx_va)} test {len(idx_te)}")
    # which layers each remaining consumer needs (owner + consumer for every variant)
    need = {c: set([c]) | {variant_spec(v, c, L)[0] for v in variants} for c in cons}
    t0 = time.time()
    for j, c in enumerate(cons):
        ii = ar[:min(T, 20000)]
        top = selection(D.z(c)[ii].float() @ R["gate"][c].T, R["bias"][c]).topk(K, 1).indices
        res["router_reproduces_capture"][c] = round(float((top[:, :, None] == D.sel[ii, c][:, None, :]).any(1).float().mean()), 4)
        for v in variants:
            S = V[v]
            h = None
            if not v.startswith("router"):
                h, rc = train_head(D, R, c, v, a, S["tr"], S["va"])
                S["heads"][c] = {k: x.half().cpu() for k, x in h.state_dict().items()}
                S["vrec"][c] = rc
            S["P"][:, c] = predict(D, R, c, v, h, S["te"])
        log(f"consumer {c:02d} ({j + 1}/{len(cons)}) {time.time() - t0:.0f} s; router reproduces {res['router_reproduces_capture'][c]}; "
            + " ".join(f"{v} val@8 {V[v]['vrec'][c]:.3f}" for v in variants if c in V[v]["vrec"]))
        if a.delete_z and a.zdir:
            still = set().union(*[need[x] for x in cons[j + 1:]]) if j + 1 < len(cons) else set()
            for l in list(D._z):
                if l not in still:
                    D._z.pop(l)
                    fn = os.path.join(a.zdir, f"z{l:02d}.f16")
                    for f in (fn, fn + ".ok"):
                        if os.path.exists(f):
                            os.remove(f)
    for v in variants:
        S = V[v]
        te = S["te"].cpu().numpy() + S["sh"]
        P = S["P"][:, cons] if a.layers else S["P"]
        cl = cons if a.layers else list(range(L))
        true = D.sel[torch.from_numpy(te).to(dev)][:, cl].cpu().numpy()
        tier = D.tier[te][:, cl].copy()
        if S["sh"] == 0 and 0 in cl:
            tier[:, cl.index(0)] = -2                          # same-token: layer 0 has no layer-ahead source
        m = metrics(P, true, tier)
        m["test_tokens"] = int(len(te)); m["shift"] = S["sh"]
        if S["vrec"]:
            m["val_recall@8_mean"] = round(float(np.mean(list(S["vrec"].values()))), 4)
        res["variants"][v] = m
        log(f"{v}: " + " ".join(f"{k}={m[k]}" for k in m if k.startswith("recall@") and ("@8/" in k or "@16/" in k or "@32/" in k)))
        np.save(os.path.join(a.out, f"pred_{v}_{tag}.npy"), S["P"])
        np.save(os.path.join(a.out, f"testidx_{v}_{tag}.npy"), te)
        if S["heads"]:
            torch.save({"heads": S["heads"], "consumers": cons, "variant": v, "hidden": a.hidden, "H": a.H, "E": E,
                        "owner_shift": variant_spec(v, 1, L)}, os.path.join(a.out, f"heads_{v}_{tag}.pt"))
    json.dump(res, open(os.path.join(a.out, f"metrics_{tag}.json"), "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
