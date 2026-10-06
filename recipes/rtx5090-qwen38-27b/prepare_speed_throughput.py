"""Resolve the SPEED-Bench throughput_8k placeholders with pinned code."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
config = sys.argv[1] if len(sys.argv) > 1 else "throughput_8k"
source = ROOT / "speed-bench-prepare.py"
spec = importlib.util.spec_from_file_location("nvidia_speed_prepare", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
revision = "454f88454792dfa3ccfd7ef15fff248efde44cd1"  # pinned nvidia/SPEED-Bench revision
original_load = module.load_dataset


def pinned_load(path, *args, **kwargs):
    if path == "nvidia/SPEED-Bench":
        kwargs["revision"] = revision
    return original_load(path, *args, **kwargs)


module.load_dataset = pinned_load
output = ROOT / f"datasets/speed-bench-resolved-{config}"
output.mkdir(parents=True, exist_ok=True)
module.prepare_data(SimpleNamespace(config=config, output_dir=output))
path = output / f"{config}.jsonl"
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
assert len({r["question_id"] for r in rows}) == len(rows)
assert not any(module.TURNS_PLACEHOLDER in m["content"] for r in rows for m in r["messages"])
manifest = {"prepared_utc": datetime.now(timezone.utc).isoformat(), "dataset_revision": revision, "config": config,
            "prepare_source": json.loads((ROOT / "speed-bench-prepare-source.json").read_text()),
            "prepare_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "prepared_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "categories": dict(Counter(r["category"] for r in rows)), "rows": len(rows),
            "turns": sum(len(r["messages"]) for r in rows), "unresolved_placeholders": 0}
(output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2), flush=True)
