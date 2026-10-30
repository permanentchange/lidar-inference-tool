#!/usr/bin/env python3
"""Run the same original PCD through CenterPoint Voxel and BEVFusion and draw BEV comparisons."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
COLORS = ['#1479b8','#d53e4f','#7b3294','#f28e2b','#34823e']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=str(REPO/'demo/data/bevfusion'))
    parser.add_argument('--centerpoint-model', default=str(REPO/'demo/centerpoint_voxel'))
    parser.add_argument('--bevfusion-model', default=str(REPO/'demo/bevfusion_exp20261028'))
    parser.add_argument('--output-dir', default=str(REPO/'outputs/comparison'))
    parser.add_argument('--frame', action='append', help='Timestamp; repeat to select frames (default: three new scenes)')
    args=parser.parse_args()
    sys.path.insert(0,str(REPO/'scripts/bevfusion'))
    from _lib.points import load_points
    from _lib.common import write
    from _lib.environment import bootstrap_native
    bootstrap_native()
    import numpy as np
    import yaml
    os.environ.setdefault('MPLCONFIGDIR',str(Path(args.output_dir)/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    data=Path(args.data_dir).resolve();output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((data/'index.json').read_text())['frames']
    selected=[f for f in catalog if f['frame_id'] in args.frame] if args.frame else [f for f in catalog if f['reason']!='original_demo']
    if not selected or (args.frame and set(args.frame)-{f['frame_id'] for f in selected}):
        raise RuntimeError('Unknown or empty frame selection')
    configs={k:yaml.safe_load((Path(p)/'model.yaml').read_text())
             for k,p in [('centerpoint_voxel',args.centerpoint_model),('bevfusion',args.bevfusion_model)]}
    baseline=configs['centerpoint_voxel']
    mapping=baseline['postprocess'].get('label_mapping') or list(range(len(baseline['model']['label_names'])))
    names=baseline['postprocess'].get('mapped_label_names') or baseline['model']['label_names']
    for config in configs.values():
        output_names=config['postprocess'].get('mapped_label_names') or config['model']['label_names']
        if output_names != names or config['model']['label_names'] != baseline['model']['label_names']:
            raise RuntimeError('Models must share original and output label definitions')
    lo=np.maximum(*[c['model']['point_cloud_range'][:3] for c in configs.values()])
    hi=np.minimum(*[c['model']['point_cloud_range'][3:] for c in configs.values()])
    roi=np.concatenate((lo,hi)).tolist()
    reports=[]
    for frame in selected:
        stamp=frame['frame_id'];target=output/stamp;target.mkdir(exist_ok=True)
        results={}
        for name,model,input_path in [('centerpoint_voxel',args.centerpoint_model,data/frame['point_cloud']),
                                      ('bevfusion',args.bevfusion_model,data/frame['manifest'])]:
            cmd=[str(REPO/'build/infer_debug_tool'),'--model-dir',model,'--input',str(input_path),
                 '--dump-dir',str(target/name),'--no-dump-internal','--no-dump-decoded']
            with (target/(name+'.log')).open('w') as log:
                subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
            results[name]=json.loads((target/name/'result.json').read_text())['objects']
        points=load_points(data/frame['point_cloud']);mask=np.all(points[:,:3]>=roi[:3],axis=1)&np.all(points[:,:3]<roi[3:],axis=1)
        gt=json.loads((data/'frames'/stamp/'annotations.json').read_text())['objects']
        def within(d):
            return all(roi[i]<=d['box'][i]<roi[i+3] for i in range(3))
        gt=[dict(d,label=mapping[d['label']]) for d in gt if d['valid'] and within(d) and mapping[d['label']] >= 0]
        views={name:[d for d in ds if within(d)] for name,ds in results.items()}
        fig,axes=plt.subplots(1,3,figsize=(15,9),sharex=True,sharey=True)
        def draw(ax,objects):
            ax.scatter(points[mask,1][::3],points[mask,0][::3],s=.25,c='0.72',rasterized=True)
            for d in objects:
                x,y,z,dx,dy,dz,yaw=d['box'];corners=np.array([[dx/2,dy/2],[dx/2,-dy/2],[-dx/2,-dy/2],[-dx/2,dy/2],[dx/2,dy/2]])
                rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
                xy=corners@rotation.T+[x,y];ax.plot(xy[:,1],xy[:,0],color=COLORS[d['label']],lw=1.4)
            ax.set_aspect('equal');ax.set_xlim(roi[4],roi[1]);ax.set_ylim(roi[0],roi[3]);ax.set_xlabel('LiDAR y (m)');ax.grid(alpha=.2)
        draw(axes[0],gt);axes[0].set_title('Ground truth (common roi)');axes[0].set_ylabel('LiDAR x (m)')
        for ax,(name,objects) in zip(axes[1:],views.items()):draw(ax,objects);ax.set_title(name+' (shared postprocess)')
        fig.suptitle(stamp+' / '+frame['reason']);fig.legend(handles=[Line2D([0],[0],color=c,label=n) for c,n in zip(COLORS,names)],loc='lower center',ncol=5)
        fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(target/'comparison.png',dpi=150);plt.close(fig)
        report=dict(frame_id=stamp,source=str(data/'frames'/stamp/'source.json'),point_cloud=str(data/frame['point_cloud']),
            common_roi=roi,common_labels=names,ground_truth_label_mapping=mapping,
            counts=dict(ground_truth=len(gt),**{k:len(v) for k,v in views.items()}),
            native_configurations={k:dict(model=v['model'],runtime=v['runtime'],postprocess=v['postprocess']) for k,v in configs.items()},
            note='Same source PCD and shared deployment policy; predictions already carry final output labels. Only ground truth is mapped for display.')
        write(target/'comparison.json',report);reports.append(report)
        print(stamp,report['counts'],flush=True)
    write(output/'report.json',dict(frames=reports))
    (output/'README.md').write_text('# Same-frame comparisons\n\n'+ '\n'.join(f"- [{r['frame_id']}]({r['frame_id']}/comparison.png)" for r in reports)+'\n')

if __name__=='__main__':main()
