"""Locally prepare, then strictly gate Phase 4 profiling/quality GPU studies.

CPU rehearsal is explicit and cannot produce a GPU-verified manifest.
Each budget is a fresh fit, including its budget-dependent schedule/readout.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from e_means.config import EMeansConfig, LloydConfig, positive_int
from e_means.gpu_protocol import (array_hash, configure_torch, cuda_identity, file_hash,
                                  source_hash, source_paths, verify_gate, write_json)
from e_means.gpu_workloads import load_workload, validate_config
from e_means.quality_protocol import variants, recipe_config


def load_config(path,root=ROOT):
    config=json.loads(Path(path).read_text())
    config.setdefault('profile_lloyd_budget',config.get('profile_budget'))
    if config.get('schema_version')!=4: raise ValueError('Phase 4 schema_version must be 4')
    base=json.loads((root/config['baseline_config']).read_text());validate_config(base)
    for key in ('repeats','warmups','calibration_iterations','sensitivity_budget','profile_budget','profile_lloyd_budget'):
        positive_int(key,config[key])
    for key in ('lloyd_budgets','emeans_budgets','seeds'):
        values=config[key]
        if not values or len(set(values))!=len(values): raise ValueError('unique nonempty '+key+' required')
        for v in values:
            if key=='seeds' and v==0 and not isinstance(v,bool): continue
            positive_int(key,v)
    if not config['dtypes'] or len(set(config['dtypes']))!=len(config['dtypes']) or any(d not in ('float32','float64') for d in config['dtypes']):
        raise ValueError('distinct FP32/FP64 dtypes required')
    import math
    factors=config['target_factors']
    if not factors or len(set(factors))!=len(factors) or any(isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in factors):
        raise ValueError('distinct positive target factors required')
    selected=config['workloads']
    if not selected or len(set(selected))!=len(selected) or set(selected)-{c['name'] for c in base['workloads']}:
        raise ValueError('unknown/duplicate/empty workload selection')
    return config,base


def jobs(config,base,n_by_case=None):
    import numpy as np
    for case in base['workloads']:
        if case['name'] not in config['workloads']: continue
        for seed,dtype in itertools.product(config['seeds'],config['dtypes']):
            n=(n_by_case or {}).get(case['name'])
            if n is None:
                n=case['n_train'] if case['kind']=='synthetic' else len(load_workload(case,seed,dtype,ROOT)[0])
            # Heterogeneous methods rotate per repeat, within each common
            # evaluation/horizon block; no fastest-repeat cherry picking.
            pending=[]
            for budget in config['lloyd_budgets']:
                for evaluation in ('every_epoch','final_only'):
                    pending.append(dict(case=case,seed=seed,dtype=dtype,method='lloyd',variant='lloyd',
                        budget=budget,evaluation=evaluation,config=asdict(LloydConfig(case['n_clusters'],seed=seed,dtype=dtype,
                            **base['common'],**dict(base['lloyd'],max_iter=budget)))))
            for variant in variants():
                core=variant['name'] in ('fixed','automatic')
                for budget,method in itertools.product(config['emeans_budgets'] if core else [config['sensitivity_budget']],('emeans','slow_ema')):
                    base_cfg=EMeansConfig(case['n_clusters'],seed=seed,dtype=dtype,**base['common'],**base['e_means'])
                    if method=='slow_ema': base_cfg=replace(base_cfg,finalize='slow_ema',**base['slow_ema'])
                    cfg,resolved=recipe_config(base_cfg,n,budget,variant)
                    for evaluation in (('every_epoch','final_only') if core else ('final_only',)):
                        pending.append(dict(case=case,seed=seed,dtype=dtype,method=method,variant=variant['name'],
                            budget=budget,evaluation=evaluation,config=asdict(cfg),resolved_recipe=resolved))
            for repeat in range(config['repeats']):
                # Independent design RNG; never consumes solver RNG streams.
                order=np.random.default_rng(np.random.SeedSequence([seed,repeat,412])).permutation(len(pending))
                for index in order: yield dict(pending[index],repeat=repeat)


def job_id(job):
    return '_'.join(str(job[k]) for k in ('seed','dtype','method','variant','budget','evaluation','repeat'))


def execute(config,base,output,manifest,device,mode):
    import numpy as np
    import torch
    from e_means.quality_experiments import timed_fit,observed_fit
    jobs_list=list(jobs(config,base,{p['workload']:p['train_shape'][0] for p in manifest['inputs']}))
    for case in base['workloads']:
        if case['name'] not in config['workloads']: continue
        for seed,dtype in itertools.product(config['seeds'],config['dtypes']):
            if source_hash(ROOT)!=manifest['source_sha256']:raise RuntimeError('source changed before cohort')
            x,test,init,digest=load_workload(case,seed,dtype,ROOT)
            expected=next(p for p in manifest['inputs'] if (p['workload'],p['seed'],p['dtype'])==(case['name'],seed,dtype))
            if digest!=expected['data_and_init_sha256']:raise RuntimeError('inputs changed after planning')
            cohort=[j for j in jobs_list if j['case']==case and j['seed']==seed and j['dtype']==dtype]
            x,test,init=(torch.as_tensor(a,device=device) for a in (x,test,init))
            key=f"{case['name']}_seed{seed}_{dtype}"
            cfg=LloydConfig(case['n_clusters'],seed=seed,dtype=dtype,**base['common'],
                            **dict(base['lloyd'],max_iter=config['calibration_iterations']))
            calibration,centers=timed_fit(cfg,'lloyd',x,test,init,device,'final_only')
            calibration.update(data_and_init_sha256=digest,role='independent predeclared target calibration',
                               source_sha256=manifest['source_sha256'],hardware=manifest['hardware'])
            np.savez_compressed(output/f'{key}_calibration.npz',**centers,init_centroids=init.cpu().numpy())
            calibration['artifact_sha256']=file_hash(output/f'{key}_calibration.npz')
            write_json(output/f'{key}_calibration.json',calibration)
            manifest['calibrations'][key]=file_hash(output/f'{key}_calibration.json')
            write_json(output/'manifest.json',manifest)
            if mode=='profile':
                cohort=[dict(seed=seed,dtype=dtype,case=case,method=m,variant='fixed',budget=config['profile_lloyd_budget'] if m=='lloyd' else config['profile_budget'],
                             evaluation='every_epoch',repeat=0) for m in ('lloyd','emeans','slow_ema')]
                for j in cohort:
                    if j['method']=='lloyd': j['config']=asdict(replace(cfg,max_iter=j['budget']))
                    else:
                        c=EMeansConfig(case['n_clusters'],seed=seed,dtype=dtype,**base['common'],**base['e_means'])
                        if j['method']=='slow_ema':c=replace(c,finalize='slow_ema',**base['slow_ema'])
                        j['config']=asdict(recipe_config(c,len(x),j['budget'],variants()[0])[0])
            warmed=set()
            for j in cohort:
                if source_hash(ROOT)!=manifest['source_sha256']: raise RuntimeError('source changed during experiment')
                constructor=LloydConfig if j['method']=='lloyd' else EMeansConfig
                cfg=constructor(**j['config'])
                # Each distinct recipe/evaluation/budget receives warmup,
                # outside the primary interval; all contenders start fresh.
                warm_key=(j['method'],j['variant'],j['budget'],j['evaluation'])
                if warm_key not in warmed:
                    for _ in range(config['warmups']): timed_fit(cfg,j['method'],x,test,init,device,j['evaluation'])
                    warmed.add(warm_key)
                record,arrays=timed_fit(cfg,j['method'],x,test,init,device,j['evaluation'])
                name=case['name']+'_'+job_id(j)
                diagnostic=observed_fit(cfg,j['method'],x,test,init,device,j['evaluation'],arrays,
                                        profile=mode=='profile',trace_path=output/f'{name}_trace.json')
                if mode=='profile': diagnostic['trace_sha256']=file_hash(output/f'{name}_trace.json')
                record.update({k:j[k] for k in ('seed','dtype','method','variant','budget','evaluation','repeat')})
                record.update(workload=case['name'],data_and_init_sha256=digest,hardware=manifest['hardware'],
                    source_sha256=manifest['source_sha256'],diagnostic=diagnostic,recipe_resolution=j.get('resolved_recipe'),
                    targets={str(factor):{metric:calibration[metric]*factor for metric in ('train_mse','test_mse')}
                             for factor in config['target_factors']},
                    status=('gpu_profiled' if mode=='profile' else 'gpu_measured') if manifest['gpu_verified'] else 'cpu_rehearsal')
                path=output/f'{name}.npz';np.savez_compressed(path,**arrays)
                record['artifact_sha256']=file_hash(path)
                write_json(output/f'{name}.json',record)
                manifest['record_sha256'][f'{name}.json']=file_hash(output/f'{name}.json')
                manifest['runs'].append(f'{name}.json');write_json(output/'manifest.json',manifest)
            del x,test,init


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['prepare','benchmark','profile'],default='prepare')
    parser.add_argument('--gate',type=Path)
    parser.add_argument('--cpu-rehearsal',action='store_true')
    args=parser.parse_args()
    config,base=load_config(args.config)
    hardware=gate=None
    gpu=args.mode!='prepare' and not args.cpu_rehearsal
    if gpu:
        if not args.gate: raise ValueError('new complete CUDA gate required; Phase 3 certificate cannot authorize changed source')
        configure_torch();hardware=cuda_identity()
        gate=json.loads(args.gate.read_text());verify_gate(gate,ROOT,hardware)
        if file_hash(args.gate.parent/'suite.xml')!=gate['junit_sha256']: raise ValueError('validation XML changed')
    elif args.cpu_rehearsal:
        configure_torch()
        hardware=dict(device='cpu',purpose='protocol rehearsal, not GPU evidence')
    plan=[]
    for case in base['workloads']:
        if case['name'] not in config['workloads']:continue
        for seed,dtype in itertools.product(config['seeds'],config['dtypes']):
            x,test,init,digest=load_workload(case,seed,dtype,ROOT)
            plan.append(dict(workload=case['name'],seed=seed,dtype=dtype,data_and_init_sha256=digest,
                             train_shape=list(x.shape),test_shape=list(test.shape)))
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'config.json',config);write_json(args.output/'baseline_config.json',base)
    planned=list(jobs(config,base,{p['workload']:p['train_shape'][0] for p in plan}))
    if args.mode=='profile':
        planned=[dict(case=case,seed=seed,dtype=dtype,method=method,variant='fixed',budget=config['profile_lloyd_budget'] if method=='lloyd' else config['profile_budget'],
                      evaluation='every_epoch',repeat=0)
                 for case in base['workloads'] if case['name'] in config['workloads']
                 for seed,dtype,method in itertools.product(config['seeds'],config['dtypes'],('lloyd','emeans','slow_ema'))]
    write_json(args.output/'planned_jobs.json',dict(jobs=planned))
    manifest=dict(schema_version=4,status='prepared' if args.mode=='prepare' else 'running',gpu_verified=gpu,
        config=config,config_sha256=file_hash(args.output/'config.json'),baseline_config_sha256=file_hash(args.output/'baseline_config.json'),
        source_sha256=source_hash(ROOT),hardware=hardware,inputs=plan,runs=[],record_sha256={},calibrations={},mode=args.mode,
        planned_measurements=len(planned),planned_jobs_sha256=file_hash(args.output/'planned_jobs.json'),
        created_utc=datetime.now(timezone.utc).isoformat(),
        matched_quality='fresh independently finalized budget grid; train/test/joint targets; test MSE never selects recipe',
        profiling='separate replay, exact arrays checked; no profiler in primary timings',
        distance_units='all-centroid point/centroid pairs, training versus metric/readout work separated')
    with zipfile.ZipFile(args.output/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths(ROOT):archive.write(path,path.relative_to(ROOT))
    manifest['source_archive_sha256']=file_hash(args.output/'source.zip')
    if gate:
        write_json(args.output/'validation_report.json',gate)
        (args.output/'validation_suite.xml').write_bytes((args.gate.parent/'suite.xml').read_bytes())
    write_json(args.output/'manifest.json',manifest)
    if args.mode=='prepare':
        print(f'Prepared {len(planned)} primary fits; ZERO GPU measurements');return
    try:
        execute(config,base,args.output,manifest,'cuda:0' if gpu else 'cpu',args.mode)
        if source_hash(ROOT)!=manifest['source_sha256']:raise RuntimeError('source changed during experiment')
        manifest.update(status='complete' if gpu else 'cpu_rehearsal_complete',completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status='failed',error=str(error));raise
    finally:write_json(args.output/'manifest.json',manifest)


if __name__=='__main__': main()
