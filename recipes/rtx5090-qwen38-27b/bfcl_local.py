"""Run the Berkeley Function Calling Leaderboard (bfcl-eval) against local OpenAI-compatible servers.

Arms: `ninfer` (fork build + selected draft-Q4 artifact, --agent-prompt-cache) and `llamacpp` (llama.cpp b11425 with
the Q4_K_M target + Q4_K_M DFlash2 draft). Both are queried through BFCL's native function-calling handler
(Chat Completions `tools`), the same path an agent client would use. Results and scores land under results/bfcl/.

Usage: python scripts/bfcl_local.py --arm ninfer --test-category simple_python,multiple --num-threads 4
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import requests

ROOT = Path(__file__).resolve().parents[1]
LINUX_ROOT = "/home/kedar/qwen38-27b-megakernel"
BFCL_ROOT = ROOT / "results" / "bfcl"
os.environ.setdefault("BFCL_PROJECT_ROOT", str(BFCL_ROOT))
BFCL_ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))

try:  # the llama.cpp arm needs the workspace's serve helper; the NInfer arms do not
    from serve import server_command, server_environment
except ImportError:  # pragma: no cover - recipe checkout without the workspace scripts
    server_command = server_environment = None

MODEL_ID = "qwen3.8-27b"
ARMS = {
    "ninfer": "ninfer-round2-fork-qwen3.8-27b-FC",
    "ninfer-stock": "ninfer-public-68c5435-qwen3.8-27b-FC",
    "llamacpp": "llamacpp-b11425-qwen3.8-27b-FC",
}
NINFER_SERVERS = {
    "ninfer": (LINUX_ROOT + "/runtimes/ninfer-round2-fork/ninfer-serve",
               LINUX_ROOT + "/models/qwen3_8_27b_gdn4_a4_draftq4.ninfer", ["--agent-prompt-cache"]),
    "ninfer-stock": (LINUX_ROOT + "/runtimes/ninfer-public-68c5435/ninfer-serve",
                     LINUX_ROOT + "/models/qwen3_8_27b_nvfp4.ninfer", []),
}


def register_models():
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

    for arm, name in ARMS.items():
        MODEL_CONFIG_MAPPING[name] = ModelConfig(
            model_name=MODEL_ID, display_name=name, url="local", org="local", license="apache-2.0",
            model_handler=OpenAICompletionsHandler, input_price=None, output_price=None,
            is_fc_model=True, underscore_to_dot=True)


def ninfer_command(arm, port, concurrency):
    server, artifact, extra = NINFER_SERVERS[arm]
    env = (f"export LD_LIBRARY_PATH={LINUX_ROOT}/runtimes/cuda-13.4.2/lib:{LINUX_ROOT}/runtimes/curl-8.22.0/lib:"
           f"{LINUX_ROOT}/runtimes/ffmpeg-8.1.3/lib:/usr/lib/wsl/lib; ")
    args = [server, artifact, "--host", "127.0.0.1", "--port", str(port), "--model-id", MODEL_ID,
            "--max-context", "65536", "--kv-capacity", "65536", "--max-concurrency", str(concurrency),
            "--kv-dtype", "bf16", "--prefill-chunk", "2048", "--device-state-slots", "4",
            "--host-context-mib", "8192", "--preserve-thinking", "--spec", "dflash2", "--draft-tokens", "7",
            "--lm-head-draft", *extra]
    return ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-c", env + "exec " + " ".join(args)], args


def llamacpp_command(port, concurrency):
    if server_command is None:
        raise RuntimeError("llama.cpp arm needs scripts/serve.py from the workspace repository")
    command = server_command("dflash", port, "runtimes/llama-b11425/llama-server.exe")
    command[command.index("--parallel") + 1] = str(concurrency)
    command += ["--alias", MODEL_ID, "--reasoning-format", "deepseek"]
    return command, command


def wait_ready(url, process, seconds=600):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited {process.returncode}")
        try:
            if requests.get(url + "/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError("server did not become ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--test-category", default="single_turn")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--port", type=int, default=9985)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--no-launch", action="store_true", help="Use an already running server on --port")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    register_models()
    from bfcl_eval._llm_response_generation import main as generation_main
    from bfcl_eval.eval_checker.eval_runner import main as evaluation_main

    model = ARMS[args.arm]
    categories = args.test_category.split(",")
    result_dir = BFCL_ROOT / f"result-{args.arm}"
    score_dir = BFCL_ROOT / f"score-{args.arm}"
    run_dir = BFCL_ROOT / "runs"
    run_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {"arm": args.arm, "model_entry": model, "categories": categories, "num_threads": args.num_threads,
              "temperature": args.temperature, "started_utc": datetime.now(timezone.utc).isoformat()}

    process = None
    log = None
    url = f"http://127.0.0.1:{args.port}"
    try:
        if not args.no_launch:
            if args.arm in NINFER_SERVERS:
                launch, shown = ninfer_command(args.arm, args.port, args.num_threads)
                environment = os.environ.copy()
            else:
                launch, shown = llamacpp_command(args.port, args.num_threads)
                environment = server_environment()
            record["server_command"] = shown
            log = (run_dir / f"{stamp}-{args.arm}-server.log").open("w", encoding="utf-8")
            process = subprocess.Popen(launch, env=environment, stdout=log, stderr=subprocess.STDOUT)
            wait_ready(url, process)
        record["models_endpoint"] = requests.get(url + "/v1/models", timeout=10).json()
        os.environ["OPENAI_BASE_URL"] = url + "/v1"
        os.environ["OPENAI_API_KEY"] = "local"
        started = time.monotonic()
        generation_main(SimpleNamespace(
            model=[model], test_category=categories, temperature=args.temperature, include_input_log=False,
            exclude_state_log=False, num_gpus=1, num_threads=args.num_threads, gpu_memory_utilization=0.9,
            backend="sglang", skip_server_setup=True, local_model_path=None, result_dir=result_dir,
            allow_overwrite=args.allow_overwrite, run_ids=False, enable_lora=False, max_lora_rank=None,
            lora_modules=None))
        record["generation_seconds"] = time.monotonic() - started
    finally:
        if process is not None:
            if args.arm in NINFER_SERVERS:
                subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "pkill", "-f", f"ninfer-serve.*--port {args.port}"],
                               check=False)
            else:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        if log is not None:
            log.close()

    evaluation_main([model], categories, result_dir, score_dir, partial_eval=True)
    scores = {}
    for path in sorted((score_dir / model).rglob("*_score.json")):
        first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        scores[path.stem.replace("BFCL_v4_", "").replace("_score", "")] = {
            "accuracy": first.get("accuracy"), "correct": first.get("correct_count"), "total": first.get("total_count")}
    record.update(scores=scores, finished_utc=datetime.now(timezone.utc).isoformat())
    (run_dir / f"{stamp}-{args.arm}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
