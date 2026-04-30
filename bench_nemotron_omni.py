"""Throughput + TTFT benchmark for Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-NVFP4 on Spark.

Hits the local OpenAI-compatible vLLM endpoint at http://localhost:8000.
Measures:
  - Single-stream decode tok/s (a single concurrent request, varying output lengths)
  - TTFT (time to first token) at batch=1, 4, 8
  - Concurrent throughput at batch=4, 8
"""
import asyncio
import json
import time
import statistics
import aiohttp
import argparse

ENDPOINT = "http://localhost:8000/v1/chat/completions"
MODEL = "AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-NVFP4"

PROMPTS = [
    "Explain in detail how a binary search tree works, including insertion, deletion, and balancing.",
    "Write a 200-word essay on the impact of quantum computing on cryptography.",
    "Describe the process of photosynthesis at the molecular level, covering both light and dark reactions.",
    "Compare and contrast Mamba2 state-space models with Transformer attention, focusing on memory complexity.",
    "Walk through the math behind backpropagation in a 3-layer MLP, with concrete equations.",
    "Explain the CAP theorem in distributed systems with three real-world database examples.",
    "Describe how RSA encryption generates and uses public/private keys, with a small worked example.",
    "Write a Python implementation of mergesort with detailed comments explaining each step.",
]


async def send_request(session, prompt, max_tokens, enable_thinking=False):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    headers = {"Content-Type": "application/json"}

    t_start = time.perf_counter()
    t_first = None
    n_tokens = 0

    async with session.post(ENDPOINT, headers=headers, json=payload) as resp:
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                if t_first is None:
                    t_first = time.perf_counter()
                n_tokens += 1  # rough: SSE-chunk count, not exact token count

    t_end = time.perf_counter()
    ttft = (t_first - t_start) if t_first is not None else None
    decode_s = (t_end - t_first) if t_first is not None else 0
    return {
        "ttft_ms": ttft * 1000 if ttft is not None else None,
        "n_chunks": n_tokens,
        "decode_s": decode_s,
        "decode_tok_s": (n_tokens / decode_s) if decode_s > 0 else 0,
        "total_s": t_end - t_start,
    }


async def bench_concurrency(session, concurrency, prompts, max_tokens, enable_thinking):
    """Send N concurrent requests, return aggregated stats."""
    selected_prompts = (prompts * ((concurrency // len(prompts)) + 1))[:concurrency]
    tasks = [send_request(session, p, max_tokens, enable_thinking) for p in selected_prompts]
    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks)
    t1 = time.perf_counter()

    total_chunks = sum(r["n_chunks"] for r in results)
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    decode_tok_s = [r["decode_tok_s"] for r in results if r["decode_tok_s"] > 0]
    elapsed = t1 - t0

    return {
        "concurrency": concurrency,
        "wall_s": elapsed,
        "total_chunks": total_chunks,
        "aggregate_chunks_per_s": total_chunks / elapsed,
        "median_ttft_ms": statistics.median(ttfts) if ttfts else None,
        "p99_ttft_ms": statistics.quantiles(ttfts, n=100)[-1] if len(ttfts) > 1 else (ttfts[0] if ttfts else None),
        "median_per_stream_tok_s": statistics.median(decode_tok_s) if decode_tok_s else 0,
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--label", type=str, default="run")
    p.add_argument("--thinking", action="store_true")
    args = p.parse_args()

    print(f"=== bench label={args.label!r}  max_tokens={args.max_tokens}  thinking={args.thinking} ===\n")

    async with aiohttp.ClientSession() as session:
        # Warmup
        print("[warmup] single short prompt...")
        await send_request(session, "Say hi.", 20, enable_thinking=args.thinking)

        # Single stream decode rate
        print("\n[single-stream]")
        single = await send_request(session, PROMPTS[0], args.max_tokens, enable_thinking=args.thinking)
        print(f"  TTFT: {single['ttft_ms']:.0f} ms")
        print(f"  decode: {single['decode_tok_s']:.1f} chunks/s ({single['n_chunks']} chunks in {single['decode_s']:.2f}s)")
        print(f"  total wall: {single['total_s']:.2f}s")

        # Concurrent throughput
        for concurrency in [4, 8]:
            print(f"\n[concurrency={concurrency}]")
            res = await bench_concurrency(session, concurrency, PROMPTS, args.max_tokens, args.thinking)
            print(f"  wall: {res['wall_s']:.2f}s")
            print(f"  aggregate: {res['aggregate_chunks_per_s']:.1f} chunks/s")
            print(f"  median per-stream decode: {res['median_per_stream_tok_s']:.1f} chunks/s")
            print(f"  median TTFT: {res['median_ttft_ms']:.0f} ms")
            if res["p99_ttft_ms"] is not None:
                print(f"  p99 TTFT: {res['p99_ttft_ms']:.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
