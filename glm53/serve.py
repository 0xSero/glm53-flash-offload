#!/usr/bin/env python3
"""OpenAI-compatible server for GLM-5.3-Flash EXL3 on one 24 GB GPU, over stock exllamav3 1.5.1 (AsyncGenerator).

Offload stack (each piece is a monkeypatch around stock exllamav3, enabled by env, see README):
  exl3_tiers.py    GLM53_ZC_VRAM=0      every routed expert lives in pinned host RAM (home copy, zero-copy readable)
  expert_cache.py  GLM53_EC=1           elastic CLOCK cache of experts in all free VRAM + prefill staging
  cpu_tier.py      GLM53_CPU_TIER=1     coldest decode misses computed by an AVX2 kernel on the host CPU
  triton_pin.py    GLM53_TRITON_PIN     fixed Triton autotune picks for the KDA (linear attention) kernels

Endpoints
  GET  /health, /v1/models, /server_info, /stats
  POST /v1/chat/completions  (stream or not; chat_template_kwargs.enable_thinking; reasoning in reasoning_content)
  POST /v1/completions       (raw prompt, stream or not)
  POST /generate, /tokenize  (SGLang-shaped, used by the benchmark and quality tools; return_logprob = teacher-forced)
No output caps: max_tokens defaults to the remaining context.

  python3 glm53/serve.py -m /models -cs 131072 --max-batch-size 8 -chunk_size 8192 -ambs 4 --host 0.0.0.0 --port 30000
"""
import argparse, asyncio, json, os, sys, time, uuid
import torch
from exllamav3 import model_init
from exllamav3.generator import AsyncGenerator, AsyncJob
from exllamav3.generator.sampler import ArgmaxSampler, ComboSampler
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI()
G = {"gen": None, "stats": {"requests": 0, "completion_tokens": 0}}


def load():
    """All model loading lives here (nothing heavy at import time)."""
    global args, model, config, cache, tok, draft_model, draft_cache, EOS, hf_tok, CTX, t_load
    ap = argparse.ArgumentParser()
    model_init.add_args(ap, cache=True, add_draft_model_args=True, default_chunk_size=2048)
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--served-name", default="glm-5.3-flash")
    ap.add_argument("--max-q-size", type=int, default=8)
    args = ap.parse_args()

    if os.environ.get("GLM53_ZC_VRAM"):
        import exl3_tiers
        exl3_tiers.install()
    if os.environ.get("GLM53_EC"):
        import expert_cache
        expert_cache.install()
    if os.environ.get("GLM53_TRITON_PIN"):
        import triton_pin
        triton_pin.install()

    t_load = time.time()
    model, config, cache, tok, draft_model, _draft_config, draft_cache = model_init.init(args)
    if os.environ.get("GLM53_ZC_VRAM"):
        print(f" -- zero-copy tier summary: {exl3_tiers.summary()}", flush=True)
    if os.environ.get("GLM53_CPU_TIER") == "1":   # routers wrapped BEFORE the cache attaches
        import cpu_tier
        cpu_tier.wrap(model)
    if os.environ.get("GLM53_EC"):
        expert_cache.attach(model)
    if os.environ.get("GLM53_CPU_TIER") == "1":   # register the pinned home copies, self-test, start the worker
        cpu_tier.start()
    t_load = time.time() - t_load
    print(f" -- loaded in {t_load:.1f} s", flush=True)

    EOS = set()
    for e in (getattr(config, "eos_token_id_list", None) or []):
        EOS.add(int(e))
    if tok.eos_token_id is not None:
        EOS.add(int(tok.eos_token_id))
    try:
        gc = json.load(open(os.path.join(args.model_dir, "generation_config.json")))
        for e in (gc.get("eos_token_id") if isinstance(gc.get("eos_token_id"), list) else [gc.get("eos_token_id")]):
            if e is not None:
                EOS.add(int(e))
    except Exception:
        pass
    EOS = sorted(EOS)

    hf_tok = None
    try:
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    except Exception as e:
        print(f" !! HF tokenizer unavailable ({e}); chat endpoint uses the built-in GLM template", flush=True)
    CTX = args.cache_size


def chat_prompt(messages, enable_thinking=True):
    if hf_tok is not None and getattr(hf_tok, "chat_template", None):
        p = hf_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=enable_thinking)
        # GLM-5.3's template ends the generation prompt with "<think>"; thinking off = empty think block
        if not enable_thinking and p.endswith("<think>"):
            p += "</think>"
        return p
    p = "[gMASK]<sop>"
    for m in messages:
        p += f"<|{m['role']}|>{m['content']}"
    return p + ("<|assistant|><think>" if enable_thinking else "<|assistant|><think></think>")


def make_sampler(sp):
    t = float(sp.get("temperature", 1.0) if sp.get("temperature") is not None else 1.0)
    if t <= 0:
        return ArgmaxSampler()
    return ComboSampler(temperature=t, top_p=float(sp.get("top_p") or 1.0), top_k=int(sp.get("top_k") or 0),
                        min_p=float(sp.get("min_p") or 0.0))


@app.on_event("startup")
async def startup():
    G["gen"] = AsyncGenerator(model=model, cache=cache, tokenizer=tok, max_batch_size=args.max_batch_size,
                              max_chunk_size=args.chunk_size, max_q_size=args.max_q_size,
                              draft_model=draft_model, draft_cache=draft_cache,
                              num_draft_tokens=args.num_draft_tokens, dynamic_draft_tokens=args.dynamic_draft,
                              draft_confidence=args.draft_confidence, ngram_match_min=args.ngram_match_min,
                              record_draft_stats=bool(draft_model is not None or args.ngram_match_min),
                              recurrent_cache_size=int(args.recurrent_cache_size * 1024 ** 3))


def ids_of(body):
    if body.get("input_ids") is not None:
        return torch.tensor([body["input_ids"]], dtype=torch.long)
    return tok.encode(body["text"], encode_special_tokens=True)


async def run_job(ids, sp, stop_strings=None):
    """Yields (text so far, completion tokens, finish or None) once per generator step."""
    gen = G["gen"]
    max_new = sp.get("max_new_tokens")
    room = CTX - ids.shape[-1] - (args.num_draft_tokens or 8) - 2
    max_new = room if max_new is None else min(int(max_new), room)
    job = AsyncJob(gen, input_ids=ids, max_new_tokens=max(1, max_new), sampler=make_sampler(sp),
                   stop_conditions=list(EOS) + list(stop_strings or []), decode_special_tokens=False)
    G["stats"]["requests"] += 1
    n, text, done = 0, "", False
    try:
        async for r in job:
            if r.get("stage") != "streaming":
                continue
            chunk = r.get("text") or ""
            tids = r.get("token_ids")
            n += int(tids.numel()) if tids is not None else (1 if chunk else 0)
            text += chunk
            fin = None
            if r.get("eos"):
                n = int(r.get("new_tokens", n))
                reason = r.get("eos_reason")
                fin = {"type": "length" if reason == "max_new_tokens" else "stop", "reason": str(reason)}
                G["stats"]["completion_tokens"] += n
                done = True
            yield text, n, fin
        done = True
    finally:
        if not done:   # client went away: stop the job, free the slot
            try:
                await job.cancel()
            except Exception:
                pass


def teacher_forced(ids, start, topn, want_ids):
    with torch.inference_mode():
        logits = model.forward(ids, {"attn_mode": "flash_attn_nc"}).float()[0]
    lp = torch.log_softmax(logits, -1)
    L = ids.shape[-1]

    def dist(row):
        v, ix = lp[row].topk(topn)
        top = [[float(a), int(b), None] for a, b in zip(v.tolist(), ix.tolist())]
        sel = [[float(lp[row, t]), int(t), None] for t in want_ids] if want_ids else []
        return top, sel
    in_top, in_ids = [], []
    for t in range(start, L):
        if t == 0:
            in_top.append(None); in_ids.append(None); continue
        a, b = dist(t - 1); in_top.append(a); in_ids.append(b)
    o_top, o_ids = dist(L - 1)
    return {"input_top_logprobs": in_top, "input_token_ids_logprobs": in_ids, "output_top_logprobs": [o_top],
            "output_token_ids_logprobs": [o_ids], "next_token": int(lp[L - 1].argmax())}


def sse(obj):
    return "data: " + json.dumps(obj) + "\n\n"


# ---- SGLang-shaped endpoints (benchmark / quality tools) ----------------------------------------------------------

@app.post("/generate")
async def generate(req: Request):
    body = await req.json()
    sp = body.get("sampling_params") or {}
    ids = ids_of(body)
    if body.get("return_logprob"):
        while len(G["gen"].jobs):
            await asyncio.sleep(0.05)
        r = teacher_forced(ids, int(body.get("logprob_start_len", 0)), int(body.get("top_logprobs_num", 20)),
                           body.get("token_ids_logprob") or [])
        meta = {k: r[k] for k in ("input_top_logprobs", "input_token_ids_logprobs", "output_top_logprobs",
                                  "output_token_ids_logprobs")}
        meta.update({"prompt_tokens": ids.shape[-1], "completion_tokens": 1, "finish_reason": {"type": "length"}})
        return JSONResponse({"text": "", "output_ids": [r["next_token"]], "meta_info": meta})
    P = ids.shape[-1]
    if body.get("stream"):
        async def events():
            g = run_job(ids, sp)
            try:
                async for text, n, fin in g:
                    yield sse({"text": text, "meta_info": {"prompt_tokens": P, "completion_tokens": n, "finish_reason": fin}})
                    if fin is None and await req.is_disconnected():
                        break
                yield "data: [DONE]\n\n"
            finally:
                await g.aclose()
        return StreamingResponse(events(), media_type="text/event-stream")
    text, n, fin = "", 0, None
    async for text, n, fin in run_job(ids, sp):
        pass
    return JSONResponse({"text": text, "meta_info": {"prompt_tokens": P, "completion_tokens": n, "finish_reason": fin}})


@app.post("/tokenize")
async def tokenize(req: Request):
    body = await req.json()
    return JSONResponse({"tokens": tok.encode(body["prompt"], encode_special_tokens=True)[0].tolist()})


# ---- OpenAI-compatible endpoints ----------------------------------------------------------------------------------

def openai_sp(body):
    return {"temperature": body.get("temperature", 1.0), "top_p": body.get("top_p", 1.0), "top_k": body.get("top_k"),
            "min_p": body.get("min_p"), "max_new_tokens": body.get("max_completion_tokens") or body.get("max_tokens")}


def stops_of(body):
    s = body.get("stop")
    return [s] if isinstance(s, str) else list(s or [])


def split_reasoning(text):
    """GLM-5.3 thinks inside <think>...</think>; the prompt already opened the block."""
    if "</think>" in text:
        r, c = text.split("</think>", 1)
        return r.replace("<think>", "").strip(), c.strip()
    return None, text.strip()


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    kw = body.get("chat_template_kwargs") or {}
    thinking = kw.get("enable_thinking", True)
    ids = tok.encode(chat_prompt(body["messages"], enable_thinking=thinking), encode_special_tokens=True)
    sp, stops = openai_sp(body), stops_of(body)
    cid, created, P = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time()), ids.shape[-1]
    if body.get("stream"):
        async def events():
            g = run_job(ids, sp, stops)
            sent, in_think = 0, thinking
            base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": args.served_name}
            yield sse(dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]))
            n, fin = 0, None
            try:
                async for text, n, fin in g:
                    new = text[sent:]
                    # hold back a possible partial "</think>" at the end of the text
                    hold = 0
                    if in_think:
                        for k in range(min(len("</think>") - 1, len(new)), 0, -1):
                            if "</think>".startswith(new[-k:]):
                                hold = k; break
                    if fin is not None:
                        hold = 0
                    new = new[:len(new) - hold] if hold else new
                    sent += len(new)
                    while new:
                        if in_think:
                            r, sep, rest = new.partition("</think>")
                            if r:
                                yield sse(dict(base, choices=[{"index": 0, "delta": {"reasoning_content": r.replace("<think>", "")}, "finish_reason": None}]))
                            if not sep:
                                break
                            in_think, new = False, rest
                        else:
                            yield sse(dict(base, choices=[{"index": 0, "delta": {"content": new}, "finish_reason": None}]))
                            break
                    if fin is None and await req.is_disconnected():
                        break
                usage = {"prompt_tokens": P, "completion_tokens": n, "total_tokens": P + n}
                yield sse(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": (fin or {}).get("type", "stop")}], usage=usage))
                yield "data: [DONE]\n\n"
            finally:
                await g.aclose()
        return StreamingResponse(events(), media_type="text/event-stream")
    text, n, fin = "", 0, None
    async for text, n, fin in run_job(ids, sp, stops):
        pass
    if thinking:
        reasoning, content = split_reasoning(text)
        if reasoning is None:   # never closed the think block (e.g. hit max_tokens)
            reasoning, content = text.replace("<think>", "").strip(), ""
    else:
        reasoning, content = None, text.strip()
    return JSONResponse({"id": cid, "object": "chat.completion", "created": created, "model": args.served_name,
                         "choices": [{"index": 0, "finish_reason": (fin or {}).get("type"),
                                      "message": {"role": "assistant", "content": content,
                                                  "reasoning_content": reasoning}}],
                         "usage": {"prompt_tokens": P, "completion_tokens": n, "total_tokens": P + n}})


@app.post("/v1/completions")
async def completions(req: Request):
    body = await req.json()
    prompt = body["prompt"] if isinstance(body["prompt"], str) else body["prompt"][0]
    ids = tok.encode(prompt, encode_special_tokens=True)
    sp, stops = openai_sp(body), stops_of(body)
    cid, created, P = "cmpl-" + uuid.uuid4().hex[:24], int(time.time()), ids.shape[-1]
    base = {"id": cid, "object": "text_completion", "created": created, "model": args.served_name}
    if body.get("stream"):
        async def events():
            g = run_job(ids, sp, stops)
            sent, n, fin = 0, 0, None
            try:
                async for text, n, fin in g:
                    if len(text) > sent:
                        yield sse(dict(base, choices=[{"index": 0, "text": text[sent:], "finish_reason": None}]))
                        sent = len(text)
                    if fin is None and await req.is_disconnected():
                        break
                yield sse(dict(base, choices=[{"index": 0, "text": "", "finish_reason": (fin or {}).get("type", "stop")}],
                               usage={"prompt_tokens": P, "completion_tokens": n, "total_tokens": P + n}))
                yield "data: [DONE]\n\n"
            finally:
                await g.aclose()
        return StreamingResponse(events(), media_type="text/event-stream")
    text, n, fin = "", 0, None
    async for text, n, fin in run_job(ids, sp, stops):
        pass
    return JSONResponse(dict(base, choices=[{"index": 0, "text": text, "finish_reason": (fin or {}).get("type")}],
                             usage={"prompt_tokens": P, "completion_tokens": n, "total_tokens": P + n}))


@app.get("/v1/models")
async def models():
    return JSONResponse({"object": "list", "data": [{"id": args.served_name, "object": "model", "owned_by": "local",
                                                     "max_model_len": CTX}]})


@app.get("/server_info")
async def server_info():
    sa = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))}
    sa.update({"context_length": CTX, "max_running_requests": args.max_batch_size, "max_total_num_tokens": CTX,
               "chunked_prefill_size": args.chunk_size, "kv_cache_dtype": args.cache_quant or "fp16",
               "engine": "exllamav3", "env": {k: v for k, v in os.environ.items() if k.startswith(("GLM53_", "EXL3_"))}})
    return JSONResponse({"context_length": CTX, "server_args": sa, "load_seconds": round(t_load, 1)})


@app.get("/stats")
async def stats():
    st = dict(G["stats"])
    if os.environ.get("GLM53_EC"):
        import expert_cache
        st["expert_cache"] = expert_cache.summary()
    if os.environ.get("GLM53_CPU_TIER") == "1":
        import cpu_tier
        st["cpu_tier"] = cpu_tier.summary()
    return JSONResponse(st)


@app.get("/health")
async def health():
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    load()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
