#!/usr/bin/env python3
"""N134 capture client: C1, sequential chat requests with natural-length completions (no max_tokens), until the server
has decoded --target tokens or the prompt list is exhausted (then it starts another pass with a new seed).
Writes one JSON line per request to --log (id, cat, pass, prompt/completion tokens, wall times, finish, text).
A per-request wall-clock guard (--req-timeout, default 45 min) only protects against a hung server; it is reported."""
import argparse, json, os, sys, time, urllib.request


def chat(url, body, timeout):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = []; reason = []; usage = None; fin = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            b = line[5:].strip()
            if b == "[DONE]":
                break
            o = json.loads(b)
            if o.get("usage"):
                usage = o["usage"]
            for ch in o.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content"):
                    first = first or time.time()
                text.append(d.get("content") or ""); reason.append(d.get("reasoning_content") or "")
                fin = ch.get("finish_reason") or fin
            if time.time() - t0 > timeout:
                fin = "client_timeout"
                break
    return dict(t0=t0, t_first=first, t_end=time.time(), usage=usage, finish=fin, text="".join(text),
                reasoning="".join(reason))


def messages_of(row, code_dir):
    if "messages" in row:
        return row["messages"]
    parts = []
    for f in row["long"]["files"]:
        p = os.path.join(code_dir, f)
        parts.append(f"File `{f}`:\n```\n{open(p, errors='replace').read()}\n```")
    return [{"role": "user", "content": "\n\n".join(parts) + "\n\n" + row["long"]["ask"]}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30334")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--panel", default="")
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--target", type=int, default=200000)
    ap.add_argument("--max-passes", type=int, default=3)
    ap.add_argument("--req-timeout", type=float, default=2700)
    ap.add_argument("--stop-file", default="")
    ap.add_argument("--only-first", type=int, default=0, help="use only the first N prompts (subset runs)")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.prompts)]
    if a.panel:
        for i, p in enumerate(json.load(open(a.panel))):
            rows.insert(i * 7, {"id": f"panel-{i}", "cat": "panel", "thinking": True,
                                "messages": [{"role": "user", "content": p["prompt"]}]})
    if a.only_first:
        rows = rows[:a.only_first]
    done = 0
    if os.path.exists(a.log):
        for l in open(a.log):
            o = json.loads(l)
            done += (o.get("usage") or {}).get("completion_tokens", 0)
    print(f"{len(rows)} prompts, {done} tokens already logged, target {a.target}", flush=True)
    log = open(a.log, "a")
    for ps in range(a.max_passes):
        for r in rows:
            if done >= a.target or (a.stop_file and os.path.exists(a.stop_file)):
                break
            body = {"model": "glm", "messages": messages_of(r, a.code_dir), "stream": True,
                    "stream_options": {"include_usage": True}, "temperature": 1.0, "top_p": 0.95,
                    "chat_template_kwargs": {"enable_thinking": bool(r.get("thinking", True))}}
            try:
                o = chat(a.url, body, a.req_timeout)
            except Exception as ex:
                o = dict(error=str(ex), t_end=time.time())
            n = (o.get("usage") or {}).get("completion_tokens", 0)
            done += n
            rec = dict(id=r["id"], cat=r["cat"], thinking=r.get("thinking", True), pass_=ps, **o)
            log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()
            dt = (o.get("t_end", 0) - (o.get("t_first") or o.get("t_end", 0)))
            print(f"[{time.strftime('%H:%M:%S')}] {r['id']} {r['cat']} pass {ps}: {n} tok in {dt:.0f} s "
                  f"({n / max(dt, 1e-3):.1f} tok/s) finish {o.get('finish')} {o.get('error', '')[:80]} | total {done}",
                  flush=True)
        if done >= a.target or (a.stop_file and os.path.exists(a.stop_file)):
            break
    # follow-up request so the server's capture writer flushes the last real request
    try:
        chat(a.url, {"model": "glm", "messages": [{"role": "user", "content": "Say OK."}], "stream": True,
                     "chat_template_kwargs": {"enable_thinking": False}}, 600)
    except Exception as ex:
        print("flush request failed:", ex)
    print(f"CAPTURE_CLIENT_DONE total {done}", flush=True)


if __name__ == "__main__":
    main()
