#!/usr/bin/env python3
"""Streaming chat client for ds4-server: time to first token and client-side decode tok/s.

usage: client.py PORT PROMPT_FILE MAX_TOKENS LABEL [PREVIOUS_ANSWER_FILE FOLLOW_UP_TEXT]
Greedy (temperature 0), reasoning off; the answer is saved next to the prompt as LABEL.answer.txt.
"""
import json, pathlib, sys, time, urllib.request

port, prompt_file, max_tokens, label = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
messages = [{"role": "user", "content": open(prompt_file).read()}]
if len(sys.argv) > 6:
    messages.append({"role": "assistant", "content": open(sys.argv[5]).read()})
    messages.append({"role": "user", "content": sys.argv[6]})
body = json.dumps({"model": "deepseek-v4.1-flash", "messages": messages, "temperature": 0,
                   "max_tokens": max_tokens, "stream": True, "reasoning_effort": "none",
                   "stream_options": {"include_usage": True}}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.monotonic(); ttft = None; usage = None; text = []
with urllib.request.urlopen(req, timeout=36000) as r:
    for line in r:
        line = line.decode(errors="replace").strip()
        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
            continue
        obj = json.loads(line[5:])
        usage = obj.get("usage") or usage
        for ch in obj.get("choices", []):
            d = ch.get("delta", {}).get("content") or ch.get("delta", {}).get("reasoning_content")
            if d:
                ttft = ttft if ttft is not None else time.monotonic() - t0
                text.append(d)
t1 = time.monotonic()
tokens = (usage or {}).get("completion_tokens", len(text))
pathlib.Path(prompt_file).with_name(f"{label}.answer.txt").write_text("".join(text))
print(json.dumps({"label": label, "ttft_s": round(ttft or 0, 2), "completion_tokens": tokens,
                  "decode_tok_s_client": round(tokens / (t1 - t0 - (ttft or 0)), 2) if ttft else 0}))
