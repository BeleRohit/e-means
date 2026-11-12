"""Integrity-checked returned-budget target comparisons and convergence plots."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from e_means.gpu_protocol import file_hash,seed_summary,write_json
from e_means.quality_protocol import grid_target,distance_work
from scripts.phase4_cuda import job_id


def summarize(root,*,allow_cpu=False):
    manifest=json.loads((root/'manifest.json').read_text())
    cpu=manifest['status']=='cpu_rehearsal_complete' and not manifest['gpu_verified']
    if not (manifest['status']=='complete' and manifest['gpu_verified']) and not (allow_cpu and cpu):
        raise ValueError('complete GPU-verified data required; CPU rehearsal requires explicit override')
    for file,key in [('config.json','config_sha256'),('baseline_config.json','baseline_config_sha256'),
                     ('source.zip','source_archive_sha256'),('planned_jobs.json','planned_jobs_sha256')]:
        if file_hash(root/file)!=manifest[key]: raise ValueError('snapshot integrity failure: '+file)
    if not cpu:
        gate=json.loads((root/'validation_report.json').read_text())
        if gate['status']!='cuda_verified' or gate['source_sha256']!=manifest['source_sha256'] or gate['hardware']!=manifest['hardware']:
            raise ValueError('validation identity mismatch')
        if file_hash(root/'validation_suite.xml')!=gate['junit_sha256']:raise ValueError('JUnit changed')
    planned=json.loads((root/'planned_jobs.json').read_text())['jobs']
    expected={j['case']['name']+'_'+job_id(j)+'.json':j for j in planned}
    if len(expected)!=len(planned) or set(manifest['runs'])!=set(expected) or len(manifest['runs'])!=len(expected):
        raise ValueError('incomplete/duplicate measurement coverage')
    inputs={(r['workload'],r['seed'],r['dtype']):r for r in manifest['inputs']}
    groups=defaultdict(list);rows=[];profiles=[];seen_calibrations={}
    for name in manifest['runs']:
        if file_hash(root/name)!=manifest['record_sha256'].get(name):raise ValueError('measurement JSON changed')
        r=json.loads((root/name).read_text());j=expected[name]
        if r['status']!=('cpu_rehearsal' if cpu else 'gpu_profiled' if manifest['mode']=='profile' else 'gpu_measured'):
            raise ValueError('unverified/mixed measurement status')
        if r['source_sha256']!=manifest['source_sha256'] or r['hardware']!=manifest['hardware']:
            raise ValueError('mixed source or hardware')
        if any(r[k]!=j[k] for k in ('seed','dtype','method','variant','budget','evaluation','repeat')):
            raise ValueError('measurement identity changed')
        if 'config' in j and r['config']!=j['config']:raise ValueError('planned solver config changed')
        cohort=(r['workload'],r['seed'],r['dtype'])
        inp=inputs[cohort]
        if r['data_and_init_sha256']!=inp['data_and_init_sha256']:raise ValueError('paired inputs changed')
        if file_hash(root/name.replace('.json','.npz'))!=r['artifact_sha256']:raise ValueError('arrays changed')
        if cohort not in seen_calibrations:
            key=f"{r['workload']}_seed{r['seed']}_{r['dtype']}"
            c=json.loads((root/f'{key}_calibration.json').read_text())
            if file_hash(root/f'{key}_calibration.json')!=manifest['calibrations'].get(key):raise ValueError('calibration JSON changed')
            if c['data_and_init_sha256']!=r['data_and_init_sha256'] or c['source_sha256']!=r['source_sha256'] or c['hardware']!=r['hardware']:
                raise ValueError('calibration identity mismatch')
            if file_hash(root/f'{key}_calibration.npz')!=c['artifact_sha256']:raise ValueError('calibration arrays changed')
            seen_calibrations[cohort]=c
        c=seen_calibrations[cohort]
        targets={str(f):{k:c[k]*f for k in ('train_mse','test_mse')} for f in manifest['config']['target_factors']}
        if r['targets']!=targets:raise ValueError('common calibration targets changed')
        if not r['diagnostic']['replay_outputs_exact']:raise ValueError('diagnostic replay not equivalent')
        if any(not np.isfinite(r[k]) or r[k]<0 for k in ('wall_seconds','train_mse','test_mse')):
            raise ValueError('invalid measurement')
        work=distance_work(inp['train_shape'][0],r['config']['n_clusters'],r['actual_epochs'],
            batch=None if r['method']=='lloyd' else r['recipe']['batch_size'],
            drop_last=False if r['method']=='lloyd' else r['config']['drop_last'])
        if r['diagnostic']['distance_pairs']['training']!=work:raise ValueError('training work count mismatch')
        if manifest['mode']=='profile':
            if not r['diagnostic']['profiling_enabled']:raise ValueError('missing profiler')
            if not cpu and not r['diagnostic']['cuda_kernel_events']:raise ValueError('no captured CUDA activity')
            if file_hash(root/name.replace('.json','_trace.json'))!=r['diagnostic']['trace_sha256']:raise ValueError('trace changed')
            profiles.append(r);continue
        key=tuple(r[k] for k in ('workload','seed','dtype','method','variant','evaluation','repeat'))
        groups[key].append(r);rows.append(r)
    hits=[]
    for key,grid in groups.items():
        for factor in manifest['config']['target_factors']:
            for metric in ('train_mse','test_mse','joint'):
                record=dict(zip(('workload','seed','dtype','method','variant','evaluation','repeat'),key))
                record.update(target_factor=factor,metric=metric,**grid_target(grid,grid[0]['targets'][str(factor)],metric))
                hits.append(record)
    comparison=[]
    lloyd={tuple(r[k] for k in ('workload','seed','dtype','evaluation','repeat','target_factor','metric')):r for r in hits if r['method']=='lloyd'}
    for hit in hits:
        if hit['method']=='lloyd':continue
        key=tuple(hit[k] for k in ('workload','seed','dtype','evaluation','repeat','target_factor','metric'))
        ref=lloyd[key]
        both=hit['reached'] and ref['reached']
        comparison.append(dict(**{k:hit[k] for k in ('workload','seed','dtype','method','variant','evaluation','repeat','target_factor','metric')},
            lloyd_reached=ref['reached'],contender_reached=hit['reached'],
            paired_lloyd_over_contender_seconds=ref['seconds']/hit['seconds'] if both else None,
            interpretation='conditional paired observed-grid ratio only when BOTH returned solutions reach identical targets'))
    return dict(status='cpu_rehearsal' if cpu else 'gpu_profiled' if profiles else 'gpu_measured',
                verified_runs=len(manifest['runs']),matched_grid=hits,paired_comparisons=comparison,
                rows=rows,profiles=profiles,
                limitation='discrete horizon grid; no interpolation, online stopping or retrospective finalization; test targets do not select recipes')


def plots(root,result):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(root/'matplotlib-cache'))
    os.environ.setdefault('XDG_CACHE_HOME',str(root/'cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[r for r in result['rows'] if r['evaluation']=='final_only' and r['variant'] in ('lloyd','fixed','automatic')]
    for case,dtype in sorted({(r['workload'],r['dtype']) for r in rows}):
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        for method,variant in sorted({(r['method'],r['variant']) for r in rows}):
            selected=[r for r in rows if (r['workload'],r['dtype'],r['method'],r['variant'])==(case,dtype,method,variant)]
            for col,xkey in enumerate(('budget','wall_seconds','training_distance_pairs')):
                x=[r['diagnostic']['distance_pairs']['training'] if xkey=='training_distance_pairs' else r[xkey] for r in selected]
                for i,metric in enumerate(('train_mse','test_mse')):
                    axes[i,col].scatter(x,[r[metric] for r in selected],s=15,alpha=.55,label=method+'/'+variant)
                    axes[i,col].set_xlabel(('Declared horizon (epochs)','Unprofiled resident wall seconds','Training distance pairs (not FLOPs)')[col])
                    axes[i,col].set_ylabel(metric);axes[i,col].grid(alpha=.2)
        axes[0,0].legend(fontsize=7)
        fig.suptitle(f'{case}/{dtype}: independently finalized budgets; individual seeds/repeats, no interpolation')
        fig.tight_layout();fig.savefig(root/f'{case}_{dtype}_quality.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--allow-cpu-rehearsal',action='store_true');parser.add_argument('--plots',action='store_true')
    args=parser.parse_args();result=summarize(args.run,allow_cpu=args.allow_cpu_rehearsal)
    # Keep full raw records in immutable per-fit files; summary stores aggregates/targets.
    raw=result.pop('rows');profiles=result.pop('profiles')
    aggregated=defaultdict(list)
    for r in raw:aggregated[tuple(r[k] for k in ('workload','dtype','method','variant','budget','evaluation'))].append(r)
    result['aggregates']=[dict(zip(('workload','dtype','method','variant','budget','evaluation'),key),
        **{metric:seed_summary(rs,metric) for metric in ('train_mse','test_mse','wall_seconds')}) for key,rs in sorted(aggregated.items())]
    write_json(args.run/'summary.json',result)
    if raw:
        keys=['workload','seed','dtype','method','variant','budget','evaluation','repeat','actual_epochs','train_mse','test_mse','wall_seconds']
        with (args.run/'metrics.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys+['training_distance_pairs','all_distance_pairs']);writer.writeheader()
            for r in raw:writer.writerow(dict(**{k:r[k] for k in keys},training_distance_pairs=r['diagnostic']['distance_pairs']['training'],all_distance_pairs=sum(r['diagnostic']['distance_pairs'].values())))
    if profiles:
        with (args.run/'profiling.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['workload','method','range','calls','host_inclusive_seconds','cuda_stream_span_seconds']);writer.writeheader()
            for r in profiles:
                for name,values in r['diagnostic']['ranges'].items():writer.writerow(dict(workload=r['workload'],method=r['method'],range=name,**values))
        write_json(args.run/'profile_operators.json',dict(profiles=[dict(workload=r['workload'],method=r['method'],operators=r['diagnostic']['operators'],synchronization_candidates=r['diagnostic']['synchronization_candidates']) for r in profiles]))
    if args.plots:plots(args.run,dict(result,rows=raw))
    print(result['status'],result['verified_runs'],'verified records')


if __name__=='__main__':main()
