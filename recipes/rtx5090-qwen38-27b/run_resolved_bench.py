"""Fixed resolved-prompt decode workload for fast A/B comparisons of one server.

Selects the first N first-turn prompts of every SPEED-Bench qualitative category from the
resolved dataset (no placeholders), sends them sequentially with the category-campaign sampling
and thinking settings, and writes a speed-bench.json / responses.json pair shaped like the
legacy 15-request harness so engine_sweep.py can validate and summarize it unchanged.
"""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "qualitative": (ROOT / "datasets/speed-bench-resolved-qualitative/qualitative.jsonl",
                    ROOT / "datasets/speed-bench-resolved-qualitative/manifest.json",
                    ("coding", "humanities", "math", "multilingual", "qa", "rag", "reasoning",
                     "roleplay", "stem", "summarization", "writing")),
    # throughput_8k workload: categories high_entropy / mixed / low_entropy.
    "throughput_8k": (ROOT / "datasets/speed-bench-resolved-throughput_8k/throughput_8k.jsonl",
                      ROOT / "datasets/speed-bench-resolved-throughput_8k/manifest.json",
                      ("high_entropy", "mixed", "low_entropy")),
}
DATASET, MANIFEST, CATEGORIES = DATASETS["qualitative"]
EXTRA_INPUTS = {"temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0, "seed": 1234,
                "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}


def select(per_category):
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]
    chosen = []
    for category in CATEGORIES:
        count = 0
        for row in rows:
            if row["category"] != category or row["messages"][0]["role"] != "user":
                continue
            chosen.append({"id": row["question_id"], "category": category,
                           "messages": [row["messages"][0]]})
            count += 1
            if count == per_category:
                break
        if count != per_category:
            raise RuntimeError(f"category {category} has fewer than {per_category} prompts")
    return chosen


def run_request(url, model, messages, osl, timeout):
    payload = {"model": model, "messages": messages, "max_tokens": osl, "stream": False}
    payload.update(EXTRA_INPUTS)
    started = time.perf_counter()
    response = requests.post(url + "/v1/chat/completions", json=payload, timeout=timeout)
    latency = time.perf_counter() - started
    response.raise_for_status()
    return response.json(), latency


def summarize(results, category):
    rows = [r for r in results if r["ok"] and (category == "overall" or r["category"] == category)]
    failed = [r for r in results if not r["ok"] and (category == "overall" or r["category"] == category)]
    draft_n = sum(r["draft_n"] for r in rows)
    accepted = sum(r["draft_n_accepted"] for r in rows)
    return {"category": category, "requests": len(rows), "turns": len(rows), "failed": len(failed),
            "avg_pred_t_s": statistics.mean(r["predicted_per_second"] for r in rows) if rows else None,
            "avg_prompt_t_s": statistics.mean(r["prompt_per_second"] for r in rows) if rows else None,
            "avg_latency": statistics.mean(r["latency_s"] for r in rows) if rows else None,
            "accept_rate": accepted / draft_n if draft_n else None,
            "accepted": accepted, "draft_n": draft_n}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--dataset", choices=DATASETS, default="qualitative")
    parser.add_argument("--osl", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--warmup", type=int, default=1, help="Excluded leading requests of the first sample")
    args = parser.parse_args()
    global DATASET, MANIFEST, CATEGORIES
    DATASET, MANIFEST, CATEGORIES = DATASETS[args.dataset]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    samples = select(args.per_category)
    for _ in range(args.warmup):
        run_request(args.url, args.model, samples[0]["messages"], args.osl, args.timeout)
    results = []
    responses = []
    for sample in samples:
        row = {"id": sample["id"], "category": sample["category"], "turns": 1, "ok": False, "error": None}
        try:
            data, latency = run_request(args.url, args.model, sample["messages"], args.osl, args.timeout)
            timings = data["timings"]
            usage = data["usage"]
            request_key = json.dumps(sample["messages"], sort_keys=True, ensure_ascii=False).encode("utf-8")
            responses.append({"request_sha256": hashlib.sha256(request_key).hexdigest(),
                              "choices": data.get("choices"), "usage": usage, "timings": timings})
            row.update({"ok": True, "latency_s": latency, "finish_reason": data["choices"][0].get("finish_reason"),
                        "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
                        "total_tokens": usage["total_tokens"],
                        "predicted_ms": timings["predicted_ms"], "predicted_per_second": timings["predicted_per_second"],
                        "prompt_ms": timings["prompt_ms"], "prompt_per_second": timings["prompt_per_second"],
                        "draft_n": timings.get("draft_n", 0), "draft_n_accepted": timings.get("draft_n_accepted", 0)})
        except Exception as error:  # noqa: BLE001 - recorded, not hidden
            row["error"] = repr(error)
        results.append(row)
        print(json.dumps({k: row.get(k) for k in ("category", "id", "completion_tokens", "predicted_per_second", "error")}), flush=True)
    summary = [summarize(results, c) for c in CATEGORIES] + [summarize(results, "overall")]
    output = {"config": {"workload": "resolved-first-turn", "dataset": args.dataset, "per_category": args.per_category, "osl": args.osl,
                         "extra_inputs": EXTRA_INPUTS, "warmup": args.warmup, "dataset_manifest": manifest,
                         "expected_samples": len(samples)},
              "selected_samples": [s["id"] for s in samples],
              "completed_samples": sum(r["ok"] for r in results),
              "failed_samples": [r for r in results if not r["ok"]],
              "results": results, "summary": summary}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    args.output.with_name("responses.json").write_text(json.dumps(responses, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    overall = summary[-1]
    print(json.dumps(overall, indent=2), flush=True)
    raise SystemExit(1 if output["failed_samples"] else 0)


if __name__ == "__main__":
    main()
