"""Offline DFlash2 draft FFN gate/up Q8 -> Q4 experiment; never overwrites a selected artifact.

Only the five draft gate/up parents [34816,5120] change format (q8_g32_fp16 -> q4_g64_fp16).
Bindings, activation policies (A16Only) and every other object are copied byte for byte.
Encoding reuses NInfer's own grouped quantizer and row-split codec, so the bytes match what the
official converter would write from the same (Q8-represented) values. Under greedy verification
the target output distribution is unchanged; only draft proposals and acceptance can change.
"""
import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

import sys as _sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO))
from tools.artifact.codecs.row_split import (decode_row_split_codes, dequantize_row_split,
                                             encode_row_split)
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorObject, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.quantization.groupwise import quantize_matrix

TARGET = 'q4_g64_fp16'
LAYOUT = 'row_split_k128_v1'
GROUP = 64
QMAX = 7


def scalar_oracle_group(values):
    """Independent NumPy re-derivation of one Q4 G64 group (canonical scale, half-even codes)."""
    maximum = np.float32(np.max(np.abs(values)))
    raw = np.float32(np.float64(maximum) / QMAX)
    scale = np.float16(raw)
    if scale == 0 and maximum > 0:
        scale = np.float16(2.0 ** -24)
    reciprocal = np.float32(1.0 / np.float64(scale)) if scale > 0 else np.float32(0)
    codes = np.clip(np.rint(values.astype(np.float32) * reciprocal), -8, 7).astype(np.int8)
    return codes, scale


def requantize(artifact, obj, rng):
    payload = artifact.read_object(obj.id)
    weight = dequantize_row_split(payload, obj.format, obj.shape, device='cpu',
                                  dtype=torch.float32)
    assert tuple(weight.shape) == tuple(obj.shape) and torch.isfinite(weight).all()
    quantized = quantize_matrix(weight, TARGET, device='cpu')
    encoded = encode_row_split(quantized.codes, quantized.scales, TARGET, obj.shape)
    back = dequantize_row_split(encoded, TARGET, obj.shape, device='cpu', dtype=torch.float32)
    diff = (back.double() - weight.double())
    relative_l2 = float(torch.sqrt((diff * diff).sum() / (weight.double() ** 2).sum()))
    max_error = float(diff.abs().max())
    # Independent scalar oracle on sampled groups, including both row-tile edges.
    scales, codes = decode_row_split_codes(encoded, TARGET, obj.shape, device='cpu')
    n, k = obj.shape
    groups = k // GROUP
    checked = 0
    rows = [0, 1, n // 2, n - 2, n - 1] + [int(r) for r in rng.integers(0, n, 27)]
    for row in rows:
        for group in [0, groups - 1] + [int(g) for g in rng.integers(0, groups, 6)]:
            values = weight[row, group * GROUP:(group + 1) * GROUP].numpy()
            oracle_codes, oracle_scale = scalar_oracle_group(values)
            got_codes = codes[row, group].numpy().astype(np.int8)
            got_scale = scales[row, group].numpy().astype(np.float16)
            assert np.array_equal(oracle_codes, got_codes), (obj.id, row, group)
            assert oracle_scale == got_scale, (obj.id, row, group, oracle_scale, got_scale)
            checked += 1
    return encoded, dict(object=obj.id, shape=list(obj.shape), source_format=obj.format,
                         source_bytes=obj.bytes, bytes=len(encoded), format=TARGET,
                         relative_l2_vs_q8=relative_l2, maximum_absolute_error=max_error,
                         scalar_oracle_groups=checked,
                         source_sha256=hashlib.sha256(payload).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--include-down', action='store_true',
                        help='also re-quantize the five draft down projections [5120,17408]')
    parser.add_argument('--include-attn-output', action='store_true',
                        help='also re-quantize the five draft attention output projections [5120,4096]')
    args = parser.parse_args()
    assert args.source.resolve() != args.output.resolve()
    assert not args.output.exists(), 'refusing to overwrite an existing artifact'
    started = time.monotonic()
    rng = np.random.default_rng(20261005)
    with Artifact(args.source) as artifact:
        bindings = deepcopy(artifact.directory.bindings)
        selected = []
        affected = []
        down_objects = set()
        attn_objects = set()
        for name, binding in bindings.items():
            if name.startswith('dflash2/layers/') and name.endswith('/mlp/gate'):
                assert 'parts' in binding and len(binding['parts']) == 1
                selected.append(binding['parts'][0]['object'])
                affected.append(name)
                affected.append(name[:-len('gate')] + 'up')
            elif args.include_down and name.startswith('dflash2/layers/') and name.endswith('/mlp/down'):
                assert 'object' in binding
                selected.append(binding['object'])
                down_objects.add(binding['object'])
                affected.append(name)
            elif args.include_attn_output and name.startswith('dflash2/layers/') and name.endswith('/attention/output'):
                assert 'object' in binding
                selected.append(binding['object'])
                attn_objects.add(binding['object'])
                affected.append(name)
        selected_set = set(selected)
        expected_objects = 5 + (5 if args.include_down else 0) + (5 if args.include_attn_output else 0)
        expected_affected = 10 + (5 if args.include_down else 0) + (5 if args.include_attn_output else 0)
        assert len(selected_set) == expected_objects and len(affected) == expected_affected, (selected, affected)
        for name in affected:
            binding = bindings[name]
            obj = binding['object'] if 'object' in binding else binding['parts'][0]['object']
            assert obj in selected_set
        for use in artifact.directory.uses:
            if use['parameter'] in affected:
                assert use['activation_policy'] == 'A16Only', use
        specs = []
        for obj in artifact.objects:
            if isinstance(obj, TensorObject):
                if obj.id in selected_set:
                    assert obj.format == 'q8_g32_fp16' and obj.layout == LAYOUT
                    expected_shape = ((5120, 17408) if obj.id in down_objects else
                                      (5120, 4096) if obj.id in attn_objects else (34816, 5120))
                    assert tuple(obj.shape) == expected_shape, obj
                    specs.append(TensorSpec(obj.id, obj.shape, TARGET, LAYOUT))
                else:
                    specs.append(TensorSpec(obj.id, obj.shape, obj.format, obj.layout))
            else:
                specs.append(ResourceSpec(obj.id, obj.bytes, obj.encoding))
        uses = deepcopy(list(artifact.directory.uses))
        provenance = deepcopy(artifact.directory.provenance)
        provenance['draft_ffn_q4_experiment'] = {
            'source': str(args.source),
            'scope': 'DFlash2 draft gate/up parents' + (' + down' if args.include_down else '') + (' + attention output' if args.include_attn_output else ''),
            'method': 'q8_g32_fp16 dequantized then NInfer grouped_absmax q4_g64_fp16',
            'activation_policy': 'A16Only unchanged'}
        report = {'source': str(args.source), 'output': str(args.output),
                  'affected_parameters': affected, 'converted': [], 'copied': {},
                  'format': TARGET, 'activation_policy': 'A16Only'}
        with ArtifactWriter(args.output, specs, components=artifact.directory.components,
                            bindings=bindings, uses=uses, metadata=artifact.directory.metadata,
                            provenance=provenance) as writer:
            for obj in artifact.objects:
                if obj.id in selected_set:
                    encoded, stats = requantize(artifact, obj, rng)
                    stats['sha256'] = hashlib.sha256(encoded).hexdigest()
                    writer.write_object(obj.id, encoded)
                    report['converted'].append(stats)
                    print(f"Converted {len(report['converted'])}/{expected_objects} {obj.id}: "
                          f"relL2={stats['relative_l2_vs_q8']:.5f} bytes {stats['source_bytes']}"
                          f"->{stats['bytes']} elapsed={time.monotonic() - started:.1f}s",
                          flush=True)
                else:
                    digest = hashlib.sha256()

                    def chunks():
                        for chunk in artifact.iter_object(obj.id):
                            digest.update(chunk)
                            yield chunk
                    writer.write_object(obj.id, chunks())
                    report['copied'][obj.id] = {'bytes': obj.bytes, 'sha256': digest.hexdigest()}
        report['source_bytes'] = artifact.file_bytes
    with Artifact(args.output) as candidate:
        report['output_bytes'] = candidate.file_bytes
        assert candidate.directory.bindings == bindings
        assert candidate.directory.uses == tuple(uses)
        expected = {**report['copied'], **{v['object']: v for v in report['converted']}}
        for obj in candidate.objects:
            digest = hashlib.sha256()
            for chunk in candidate.iter_object(obj.id):
                digest.update(chunk)
            assert digest.hexdigest() == expected[obj.id]['sha256'], obj.id
    with args.output.open('rb') as stream:
        report['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    report['size'] = args.output.stat().st_size
    report['elapsed_seconds'] = time.monotonic() - started
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('source_bytes', 'output_bytes', 'sha256',
                                              'elapsed_seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
