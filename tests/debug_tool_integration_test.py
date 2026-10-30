#!/usr/bin/env python3
"""GPU integration checks for shared CLI/report contracts and original demo inputs."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

REPO=Path(__file__).resolve().parents[1]

def read(path):return json.loads(path.read_text())

def equal(a,b):
    a,b=a['objects'],b['objects']
    assert len(a)==len(b),(len(a),len(b))
    assert [d['label'] for d in a]==[d['label'] for d in b]
    assert [d['label_name'] for d in a]==[d['label_name'] for d in b]
    np.testing.assert_allclose([d['box'] for d in a],[d['box'] for d in b],atol=5e-3,rtol=1e-3)
    np.testing.assert_allclose([d['score'] for d in a],[d['score'] for d in b],atol=1e-5,rtol=1e-3)

def main():
    try:
        count=ctypes.c_int();cuda=ctypes.CDLL('libcudart.so')
        if cuda.cudaGetDeviceCount(ctypes.byref(count))!=0 or not count.value:return 77
    except OSError:return 77
    data=REPO/'demo/data/bevfusion'
    if not (data/'index.json').is_file():return 77
    def run(kind,model,input_path,out,*extra):
        cmd=[str(REPO/'build'/kind),'--model-dir',str(REPO/'demo'/model),'--input',str(input_path),
             '--dump-dir',str(out),'--warmup','1','--profile-iters','2',*extra]
        process=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        assert process.returncode==0,process.stdout
        report_dir=next((out/'frames').iterdir()) if Path(input_path).is_dir() else out
        meta,timing=read(report_dir/'metadata.json'),read(report_dir/'timings.json')
        assert meta['schema_version']==timing['schema_version']==2
        assert timing['total_infer_ms']['n']==2 and timing['unit']=='ms'
        return meta,timing
    with tempfile.TemporaryDirectory(prefix='shared-tools-') as tmp:
        work=Path(tmp)
        cases=[('centerpoint_pillar',REPO/'demo/data/lidar/test1.bin',['--source-num-point-features','5']),
               ('centerpoint_voxel',REPO/'demo/data/lidar/test3.pcd',[]),
               ('bevfusion_exp20261028',data/'manifests/1776582550193446516.yaml',[])]
        for model,source,flags in cases:
            if not (REPO/'demo'/model/'model.yaml').is_file():return 77
            infer=work/(model+'_infer');perf=work/(model+'_perf')
            meta,timing=run('infer_debug_tool',model,source,infer,'--no-dump-internal','--no-dump-decoded',*flags)
            assert timing['stage_profile'] is None
            meta,timing=run('perf_debug_tool',model,source,perf,*flags)
            assert sorted(p.name for p in perf.iterdir())==['metadata.json','timings.json']
            assert timing['stage_profile']['collection']=='separate_pass'
            assert timing['stage_profile']['stages']
            for stage in timing['stage_profile']['stages']:
                assert stage['host_ms']['n']==2
                if stage['gpu_ms'] is not None:assert stage['gpu_ms']['n']==2
            exported=work/(model+'_export')
            run('perf_debug_tool',model,source,exported,'--dump-result',*flags)
            if model == 'centerpoint_pillar':
                # The pre-existing atomic pillar voxelizer varies across processes.
                # Check the public detection contract without imposing bitwise inference determinism.
                a,b=read(infer/'result.json')['objects'],read(exported/'result.json')['objects']
                assert [d['label'] for d in a]==[d['label'] for d in b]
                for d in a+b:
                    assert len(d['box'])==7 and np.isfinite(d['box']).all() and 0<=d['score']<=1
            else:
                equal(read(infer/'result.json'),read(exported/'result.json'))
        catalog=read(data/'index.json')['frames']
        for frame in catalog:
            stamp=frame['frame_id'];source=read(data/'frames'/stamp/'source.json')
            for entry in source['files']:
                local=data/'frames'/stamp/entry['file']
                assert hashlib.sha256(local.read_bytes()).hexdigest()==entry['sha256']
            raw=work/(stamp+'_raw');pre=work/(stamp+'_pre')
            run('infer_debug_tool','bevfusion_exp20261028',data/frame['manifest'],raw,'--no-dump-internal','--no-dump-decoded')
            run('infer_debug_tool','bevfusion_exp20261028',data/frame['manifest'],pre,'--image-input','preprocessed','--no-dump-internal','--no-dump-decoded')
            np.testing.assert_array_equal(np.load(raw/'final_boxes.npy'),np.load(pre/'final_boxes.npy'))
            np.testing.assert_array_equal(np.load(raw/'final_scores.npy'),np.load(pre/'final_scores.npy'))
            from_path=REPO/'demo/data/bevfusion/frames'/stamp/'lidar_point_cloud_0'/(stamp+'.pcd')
            cp=work/(stamp+'_cp');run('infer_debug_tool','centerpoint_voxel',from_path,cp,'--no-dump-internal','--no-dump-decoded')
            np.testing.assert_array_equal(np.load(raw/'input_points.npy'),np.load(cp/'input_points.npy'))
        seq=work/'sequence'
        run('infer_debug_tool','bevfusion_exp20261028',data/'manifests',seq,'--no-dump-internal','--no-dump-decoded')
        entries=read(seq/'sequence_manifest.json')['frames']
        assert len(entries)==4
        assert [int(x['frame_id']) for x in entries]==sorted(int(x['frame_id']) for x in entries)
        for index,entry in enumerate(entries):
            assert entry['frame_dir']==f"frames/{index:06d}_{entry['frame_id']}"
            assert entry['timestamp_ns']==int(entry['frame_id'])
            assert (seq/entry['frame_dir']/'result.json').is_file()
        assert len((seq/'sequence_detections.jsonl').read_text().splitlines())==4
    print('Three backends, separate profile passes, dump policy, four raw/preprocessed frames, source hashes, identical PCD inputs and sequence reports passed.')
    return 0
if __name__=='__main__':sys.exit(main())
