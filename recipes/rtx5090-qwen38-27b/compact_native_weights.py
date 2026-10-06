"""Offline FP8 -> NVFP4 experiment; never overwrites the selected artifact.

Uses NInfer's validated container writer and exact scalar format definitions.
The quantizer is local NumPy code, including a small independent scalar oracle.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO))
from tools.artifact.formats import decode_e2m1_word, decode_e4m3fn_word
from tools.artifact.layouts import block_scale_geometry, row_scale_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject, TensorSpec, ResourceSpec
from tools.artifact.writer import ArtifactWriter

FP8 = np.array([decode_e4m3fn_word(i) for i in range(256)], dtype=np.float32)
FP4 = np.array([decode_e2m1_word(i) for i in range(16)], dtype=np.float32)
FP8_MID = (FP8[:126] + FP8[1:127]) * np.float32(.5)
FP4_MID = (FP4[:7] + FP4[1:8]) * np.float32(.5)
CLIPS = tuple(np.float32(v) for v in (.80, .85, .90, .95, 1.00))


def nearest_even(value, midpoints):
    index = np.searchsorted(midpoints, value, side='left')
    at_midpoint = (index < len(midpoints)) & (value == midpoints[np.minimum(index, len(midpoints)-1)])
    return (index + (at_midpoint & ((index & 1) != 0))).astype(np.uint8)


def encode_groups(values, divisor):
    """FP32 groups [...,16]; choose an E4M3FN scale, then signed E2M1 words."""
    scaled = values * divisor
    absolute = np.abs(scaled)
    base_scale = absolute.max(axis=-1) / np.float32(6)
    best_error = np.full(base_scale.shape, np.inf, dtype=np.float32)
    best_codes = np.zeros(values.shape, dtype=np.uint8)
    best_scales = np.zeros(base_scale.shape, dtype=np.uint8)
    signs = np.signbit(values).astype(np.uint8) * 8
    for clip in CLIPS:
        scale_words = nearest_even(base_scale * clip, FP8_MID)
        scales = FP8[scale_words]
        ratio = np.divide(absolute, scales[..., None],
                          out=np.zeros_like(absolute), where=scales[..., None] != 0)
        codes = nearest_even(ratio, FP4_MID) | signs
        decoded = FP4[codes] * scales[..., None]
        error = np.sum((decoded - scaled)**2, axis=-1, dtype=np.float32)
        improve = error < best_error
        best_error = np.where(improve, error, best_error)
        best_scales = np.where(improve, scale_words, best_scales)
        best_codes = np.where(improve[..., None], codes, best_codes)
    return best_codes, best_scales


def scalar_group(values, divisor):
    # Exhaustive scalar nearest-code decisions, independent of vector midpoint search.
    scaled = [np.float32(x * divisor) for x in values]
    base = np.float32(max(abs(float(x)) for x in scaled) / 6)
    best = None
    for clip in CLIPS:
        target = np.float32(base * clip)
        sw = min(range(127), key=lambda i: (abs(float(FP8[i])-float(target)), i & 1, i))
        codes = []
        for original, x in zip(values, scaled):
            ratio = np.float32(abs(x) / FP8[sw]) if sw else np.float32(0)
            code = min(range(8), key=lambda i: (abs(float(FP4[i])-float(ratio)), i & 1, i))
            codes.append(code | (8 if np.signbit(original) else 0))
        diff = np.array([np.float32(FP4[c] * FP8[sw] - x) for c,x in zip(codes,scaled)], dtype=np.float32)
        error = np.sum(diff**2, dtype=np.float32)
        if best is None or error < best[0]: best = (error, codes, sw)
    return np.array(best[1],dtype=np.uint8), best[2]


def self_test():
    for table, mid in ((FP8[:127],FP8_MID),(FP4[:8],FP4_MID)):
        assert np.array_equal(nearest_even(table,mid),np.arange(len(table),dtype=np.uint8))
        assert np.array_equal(nearest_even(mid,mid),np.array([i if i%2==0 else i+1 for i in range(len(mid))],dtype=np.uint8))
    rng=np.random.default_rng(1729)
    values=rng.normal(size=(128,16)).astype(np.float32)
    values[0]=0
    values[1]=-np.float32(0)
    values[2]=FP4
    for divisor in (np.float32(.03125), np.float32(1), np.float32(129)):
        codes,scales=encode_groups(values,divisor)
        for i,row in enumerate(values):
            expected,scale=scalar_group(row,divisor)
            assert np.array_equal(codes[i],expected) and int(scales[i])==scale,(divisor,i)
    assert all(decode_e2m1_word(i)==float(FP4[i]) for i in range(16))
    print('PASS: all finite magnitude words, nearest-even ties, signed zero and 384 scalar-oracle groups',flush=True)


def bound_objects(binding):
    if 'object' in binding: return {binding['object']}
    return {part['object'] for part in binding['parts']}


def activation_auxiliaries(artifact, bindings, affected):
    """Conservative power-of-two divisors for BF16 RMSNorm mixer inputs.

    |RMSNorm(x)[i]| <= sqrt(H)*|1+norm_weight[i]| before final BF16 rounding.
    A 1% margin covers rounding; the power-of-two divisor keeps the corresponding
    K16 max/6 scale below E4M3FN's 448. This is a bound, not data calibration.
    """
    result={}
    for name in affected:
        layer=int(name.split('/')[2])
        identifier=f'experiment/gdn4/activation_divisor/{layer}'
        if identifier in result: continue
        norm=artifact.object(bindings[f'text/layers/{layer}/input_norm']['object'])
        assert norm.format=='bf16' and norm.shape==(5120,)
        weight=(np.frombuffer(artifact.read_object(norm.id),dtype='<u2').astype(np.uint32)<<16).view(np.float32)
        maximum=float(np.max(np.abs(np.float32(1)+weight)))
        bound=math.sqrt(5120)*maximum*1.01
        divisor=2.0**math.floor(math.log2(2688/bound)) if bound else 1.0
        result[identifier]={'object':identifier,'layer':layer,'divisor':divisor,
                            'max_norm_multiplier':maximum,'activation_bound':bound,
                            'bytes':4,'sha256':hashlib.sha256(struct.pack('<f',divisor)).hexdigest()}
    return result


def quantize(artifact, obj):
    n,k=obj.shape
    old=row_scale_geometry(obj.format,obj.shape)
    new=block_scale_geometry('nvfp4',obj.shape)
    payload=artifact.read_object(obj.id)
    words=np.frombuffer(payload,dtype=np.uint8,count=n*k).reshape(n,k)
    scales=(np.frombuffer(payload,dtype='<u2',count=n,offset=old.scale_plane_offset).astype(np.uint32)<<16).view(np.float32)
    assert np.isfinite(scales).all() and (scales>=0).all()
    maximum=0.0
    for begin in range(0,n,128):
        w=FP8[words[begin:begin+128]]*scales[begin:begin+128,None]
        assert np.isfinite(w).all()
        maximum=max(maximum,float(np.abs(w).max()))
    divisor=np.float32(2688.0/maximum) if maximum else np.float32(1)
    packed=np.empty((n,k//2),dtype=np.uint8)
    natural_scales=np.empty((n,k//16),dtype=np.uint8)
    sum_error=sum_weight=0.0
    max_error=0.0
    for begin in range(0,n,128):
        w=FP8[words[begin:begin+128]]*scales[begin:begin+128,None]
        codes,scale_words=encode_groups(w.reshape(128,k//16,16),divisor)
        flat=codes.reshape(128,k)
        packed[begin:begin+128]=flat[:,::2] | (flat[:,1::2]<<4)
        natural_scales[begin:begin+128]=scale_words
        decoded=(FP4[codes]*FP8[scale_words][...,None]/divisor).reshape(128,k)
        diff=decoded.astype(np.float64)-w.astype(np.float64)
        sum_error+=float(np.sum(diff*diff))
        sum_weight+=float(np.sum(w.astype(np.float64)**2))
        max_error=max(max_error,float(np.abs(diff).max()))
        # Deterministic sample across all row tiles against the independent scalar oracle.
        scalar_codes,scalar_scale=scalar_group(w.reshape(128,k//16,16)[0,0],divisor)
        assert np.array_equal(codes[0,0],scalar_codes) and int(scale_words[0,0])==scalar_scale
    stored=natural_scales.reshape(n//128,4,32,k//64,4).transpose(0,3,2,1,4).copy().reshape(-1)
    encoded=bytearray(new.payload_bytes)
    encoded[:new.code_plane_bytes]=packed.tobytes()
    encoded[new.scale_plane_offset:new.weight_divisor_offset]=stored.tobytes()
    encoded[new.weight_divisor_offset:]=struct.pack('<f',float(divisor))
    # Independent scalar swizzle/address checks, including tile and row-group edges.
    for row in (0,1,31,32,63,64,95,96,127,128,n-1):
        for group in (0,1,3,4,k//16-1):
            index=((((row//128*(k//64)+group//4)*32+row%32)*4+(row%128)//32)*4+group%4)
            sw=encoded[new.scale_plane_offset+index]
            assert sw==int(natural_scales[row,group])
            for lane in (0,1,14,15):
                column=group*16+lane
                code=(encoded[row*(k//2)+column//2] >> ((column%2)*4)) & 15
                expected=decode_e2m1_word(code)*decode_e4m3fn_word(sw)/float(divisor)
                assert np.isfinite(expected)
                assert code==int((packed[row,column//2] >> ((column%2)*4)) & 15)
    return bytes(encoded),dict(object=obj.id,shape=list(obj.shape),source_bytes=obj.bytes,
        bytes=new.payload_bytes,weight_divisor=float(divisor),relative_l2=(sum_error/sum_weight)**.5,
        maximum_absolute_error=max_error,source_sha256=hashlib.sha256(payload).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    self_test()
    if args.self_test: return
    if not all((args.source,args.output,args.report)): parser.error('source, output and report required')
    assert args.source.resolve()!=args.output.resolve()
    started=time.monotonic()
    with Artifact(args.source) as artifact:
        bindings=deepcopy(artifact.directory.bindings)
        selected=set()
        affected=[]
        for name,binding in bindings.items():
            if name.startswith('text/layers/') and name.endswith(('/gdn/query','/gdn/key','/gdn/value','/gdn/z')):
                selected.update(bound_objects(binding)); affected.append(name)
        assert len(selected)==48 and len(affected)==192
        auxiliaries=activation_auxiliaries(artifact,bindings,affected)
        specs=[]
        for obj in artifact.objects:
            if isinstance(obj,TensorObject):
                if obj.id in selected:
                    assert obj.format=='fp8_e4m3fn_row_bf16' and obj.shape==(16384,5120)
                    specs.append(TensorSpec(obj.id,obj.shape,'nvfp4','block_scale_k16_m128x4_v1'))
                else: specs.append(TensorSpec(obj.id,obj.shape,obj.format,obj.layout))
            else: specs.append(ResourceSpec(obj.id,obj.bytes,obj.encoding))
        specs.extend(TensorSpec(name,(),'fp32','contiguous_le_v1') for name in auxiliaries)
        uses=deepcopy(list(artifact.directory.uses))
        for use in uses:
            if use['parameter'] in affected:
                use['activation_policy']='AllowA4'
                layer=int(use['parameter'].split('/')[2])
                use['auxiliaries']={'activation_input_divisor':{'object':f'experiment/gdn4/activation_divisor/{layer}'}}
        provenance=deepcopy(artifact.directory.provenance)
        provenance['compact_experiment']={'source':str(args.source),'scope':'48 GDN input parents only','method':'FP8 to NVFP4 K16 five-scale MSE search','clip_candidates':[float(x) for x in CLIPS]}
        report={'source':str(args.source),'output':str(args.output),'affected_parameters':affected,
                'converted':[],'copied':{},'format':'nvfp4','activation_policy':'AllowA4',
                'activation_divisors':auxiliaries}
        with ArtifactWriter(args.output,specs,components=artifact.directory.components,
                bindings=bindings,uses=uses,metadata=artifact.directory.metadata,provenance=provenance) as writer:
            for obj in artifact.objects:
                if obj.id in selected:
                    encoded,stats=quantize(artifact,obj)
                    stats['sha256']=hashlib.sha256(encoded).hexdigest()
                    writer.write_object(obj.id,encoded)
                    report['converted'].append(stats)
                    print(f"Converted {len(report['converted'])}/48 {obj.id}: relL2={stats['relative_l2']:.6f}, elapsed={time.monotonic()-started:.1f}s",flush=True)
                else:
                    digest=hashlib.sha256()
                    def chunks():
                        for chunk in artifact.iter_object(obj.id):
                            digest.update(chunk)
                            yield chunk
                    writer.write_object(obj.id,chunks())
                    report['copied'][obj.id]={'bytes':obj.bytes,'sha256':digest.hexdigest()}
            for name,info in auxiliaries.items():
                writer.write_object(name,struct.pack('<f',info['divisor']))
        report['source_bytes']=artifact.file_bytes
    with Artifact(args.output) as candidate:
        report['output_bytes']=candidate.file_bytes
        assert candidate.directory.bindings==bindings
        assert candidate.directory.uses==tuple(uses)
        # Read back every object; unchanged payload hashes prove exact copying.
        expected={**report['copied'],**{v['object']:v for v in report['converted']},**auxiliaries}
        for obj in candidate.objects:
            digest=hashlib.sha256()
            for chunk in candidate.iter_object(obj.id): digest.update(chunk)
            assert digest.hexdigest()==expected[obj.id]['sha256'],obj.id
    with args.output.open('rb') as stream: report['sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    report['elapsed_seconds']=time.monotonic()-started
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('source_bytes','output_bytes','sha256','elapsed_seconds')},indent=2),flush=True)

if __name__=='__main__': main()
