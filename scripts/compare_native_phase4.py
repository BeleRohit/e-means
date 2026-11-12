"""Optional pinned authors native-Torch comparison, same device/FP32.

Never invokes optional Triton/CuTe kernels. CUDA mode requires a new full gate.
Default author BF16 casting is explicitly disabled through its pinned typed
input hook. Revival is disabled on both sides to isolate backend arithmetic.
These settings are NOT the paper/default authors GPU benchmark recipe.
"""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import sys
import types
from time import perf_counter
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.fetch_reference import COMMIT,HASHES,NATIVE_HASHES
from e_means.gpu_protocol import configure_torch,cuda_identity,file_hash,source_hash,source_paths,verify_gate,write_json
from e_means.gpu_workloads import load_workload


def load_native(root):
    for filename,digest in dict({'base.py':HASHES['base.py']},**NATIVE_HASHES).items():
        if file_hash(root/filename)!=digest:raise ValueError('pinned native reference mismatch: '+filename)
    name='_phase4_native_reference'
    package=types.ModuleType(name);package.__path__=[str(root)];sys.modules[name]=package
    def load(filename):
        spec=importlib.util.spec_from_file_location(name+'.'+filename,root/(filename+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module
    backend=load('torch_backend');model=load('e_means_torch')
    return model,backend


def authors_fit(module,x,init,cfg):
    import torch
    model=module.Emeans(x.shape[1],len(init),nepochs=cfg.epochs,batch_size=cfg.batch_size,alpha0=cfg.alpha0,
        seed=cfg.seed,assignment_backend='native',unused_threshold=0,max_points_per_centroid=None,
        assignment_chunk_size=cfg.assignment_chunk_size,centroid_chunk_size=cfg.centroid_chunk_size)
    # Explicit compatibility adapter to the pinned API. CUDA default Emeans
    # otherwise casts FP32 input to BF16 even with the native backend selected.
    model._bind_input_dtype(torch.float32,cast=False)
    model._prepare_input(x)
    model.begin(len(x),init)
    rng=np.random.default_rng(np.random.SeedSequence(cfg.seed).spawn(3)[1])
    for _ in range(cfg.epochs):
        order=torch.as_tensor(rng.permutation(len(x)),device=x.device)
        for offset in range(0,len(x),cfg.batch_size):
            batch=x[order[offset:offset+cfg.batch_size]]
            model.update(model.accumulate(batch))
        model.end_epoch()
    result=model.finalize()
    if model._input_dtype!=torch.float32 or model._resolved_assignment_backend!='native':
        raise RuntimeError('reference representation/backend changed')
    return result,model.recipe


def authors_assignment(backend,x,c,row_chunk,center_chunk):
    import torch
    engine=backend.TorchBackend(center_chunk)
    distance=torch.empty(len(x),device=x.device,dtype=x.dtype)
    labels=torch.empty(len(x),device=x.device,dtype=torch.long)
    for offset in range(0,len(x),row_chunk):
        d,ids=engine.assign(x[offset:offset+row_chunk],c)
        distance[offset:offset+len(d)]=d;labels[offset:offset+len(ids)]=ids
    return labels,distance


def summarize(root,*,allow_cpu=False):
    from collections import defaultdict
    from e_means.gpu_protocol import seed_summary
    m=json.loads((root/'manifest.json').read_text())
    cpu=not m['gpu_verified'] and m['status']=='cpu_rehearsal_complete'
    if not (m['gpu_verified'] and m['status']=='complete') and not(allow_cpu and cpu):raise ValueError('complete GPU comparison required; CPU override explicit')
    for file,key in [('config.json','config_sha256'),('baseline_config.json','baseline_config_sha256'),('source.zip','source_archive_sha256')]:
        if file_hash(root/file)!=m[key]:raise ValueError('native comparison snapshot changed')
    if not cpu:
        gate=json.loads((root/'validation_report.json').read_text())
        if gate['status']!='cuda_verified' or gate['source_sha256']!=m['source_sha256'] or gate['hardware']!=m['hardware'] or file_hash(root/'validation_suite.xml')!=gate['junit_sha256']:
            raise ValueError('native comparison gate mismatch')
    config=json.loads((root/'config.json').read_text())
    expected={f'{case}_seed{seed}_repeat{rep}.json' for case in config['workloads'] for seed in config['seeds'] for rep in range(config['repeats'])}
    if set(m['runs'])!=expected or len(m['runs'])!=len(expected):raise ValueError('incomplete native comparison')
    groups=defaultdict(list);parity=[]
    for name in m['runs']:
        if file_hash(root/name)!=m['record_sha256'][name]:raise ValueError('native record changed')
        r=json.loads((root/name).read_text())
        if r['source_sha256']!=m['source_sha256'] or r['hardware']!=m['hardware'] or r['status']!=('cpu_rehearsal' if cpu else 'gpu_measured'):
            raise ValueError('mixed native comparison identity')
        if file_hash(root/name.replace('.json','.npz'))!=r['artifact_sha256']:raise ValueError('native arrays changed')
        parity.append({k:r[k] for k in ('workload','seed','repeat','max_abs_center_difference','max_abs_distance_difference','exact_assignment_labels')})
        for method,fit in r['fits'].items():
            if any(not np.isfinite(fit[k]) or fit[k]<0 for k in ('wall_seconds','train_mse','test_mse')):raise ValueError('invalid native measurement')
            groups[(r['workload'],method)].append(dict(fit,seed=r['seed'],assignment_wall_seconds=r['assignment_wall_seconds'][method]))
    return dict(status='cpu_rehearsal' if cpu else 'gpu_measured',paired_records=len(expected),parity=parity,
        summary=[dict(workload=key[0],method=key[1],**{metric:seed_summary(rs,metric) for metric in ('wall_seconds','assignment_wall_seconds','train_mse','test_mse')}) for key,rs in sorted(groups.items())],
        caveat='raw backend timings and precision differences; no matched-quality full-fit speedup ratio inferred')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/phase4_pilot.json')
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',type=Path);p.add_argument('--cpu-rehearsal',action='store_true')
    args=p.parse_args()
    from scripts.phase4_cuda import load_config
    from e_means.config import EMeansConfig
    from e_means.quality_experiments import make_model,synchronize
    from e_means.torch_backend import assign_nearest
    import torch
    configure_torch();gpu=not args.cpu_rehearsal
    hardware=dict(device='cpu',purpose='compatibility rehearsal')
    gate=None
    if gpu:
        if args.gate is None:raise ValueError('full current CUDA gate required')
        hardware=cuda_identity();gate=json.loads(args.gate.read_text());verify_gate(gate,ROOT,hardware)
        if file_hash(args.gate.parent/'suite.xml')!=gate['junit_sha256']:raise ValueError('validation XML changed')
    module,backend=load_native(args.reference)
    config,base=load_config(args.config);device='cuda:0' if gpu else 'cpu'
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'config.json',config);write_json(args.output/'baseline_config.json',base)
    with zipfile.ZipFile(args.output/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths(ROOT):archive.write(path,path.relative_to(ROOT))
    manifest=dict(status='running',gpu_verified=gpu,source_sha256=source_hash(ROOT),hardware=hardware,
        reference_commit=COMMIT,reference_hashes=dict({'base.py':HASHES['base.py']},**NATIVE_HASHES),runs=[],record_sha256={},
        source_archive_sha256=file_hash(args.output/'source.zip'),config_sha256=file_hash(args.output/'config.json'),
        baseline_config_sha256=file_hash(args.output/'baseline_config.json'),repeats=config['repeats'],
        settings=dict(dtype='float32',assignment_backend='native',initial_mass=2**-16,revival_threshold=0,
            tails='keep',epochs=config['sensitivity_budget'],batch_size=256,alpha0=.2,
            representation_override='authors _bind_input_dtype(torch.float32, cast=False); disables default BF16 cast',
            comparison='own final-only full fit versus authors caller lifecycle; host bookkeeping differs; not isolated kernel speedup',
            assignment_comparison='same row/centroid chunks; no precomputed norms; norm/dot versus direct differences; precision differences reported'))
    if gate:
        write_json(args.output/'validation_report.json',gate)
        (args.output/'validation_suite.xml').write_bytes((args.gate.parent/'suite.xml').read_bytes())
    write_json(args.output/'manifest.json',manifest)
    try:
        for case in base['workloads']:
            if case['name'] not in config['workloads']:continue
            for seed in config['seeds']:
                xx,tt,ii,digest=load_workload(case,seed,'float32',ROOT)
                x,test,init=(torch.as_tensor(a,device=device) for a in (xx,tt,ii))
                cfg=EMeansConfig(case['n_clusters'],epochs=config['sensitivity_budget'],batch_size=256,alpha0=.2,
                    seed=seed,dtype='float32',drop_last=False,initial_mass=2**-16,revival_threshold=0,**base['common'])
                methods=('ours_torch','authors_native')
                for method in methods:
                    if method=='ours_torch':make_model(cfg,'emeans',device,'final_only').fit(x,init_centroids=init,eval_X=test)
                    else:authors_fit(module,x,init,cfg)
                for repeat in range(config['repeats']):
                    results={}
                    for method in methods if (seed+repeat)%2==0 else methods[::-1]:
                        if source_hash(ROOT)!=manifest['source_sha256']:raise RuntimeError('source changed')
                        synchronize(device);start=perf_counter()
                        if method=='ours_torch':
                            model=make_model(cfg,'emeans',device,'final_only').fit(x,init_centroids=init,eval_X=test)
                            centers=model.cluster_centers_;recipe=model.recipe_
                        else:centers,recipe=authors_fit(module,x,init,cfg)
                        train=assign_nearest(x,centers,chunk_size=cfg.assignment_chunk_size,centroid_chunk_size=cfg.centroid_chunk_size)[1].double().mean().item()
                        loss=assign_nearest(test,centers,chunk_size=cfg.assignment_chunk_size,centroid_chunk_size=cfg.centroid_chunk_size)[1].double().mean().item()
                        synchronize(device);elapsed=perf_counter()-start
                        results[method]=dict(centroids=centers.cpu().numpy(),recipe=recipe,wall_seconds=elapsed,train_mse=train,test_mse=loss)
                        if method=='ours_torch':del model
                    assigned={}
                    for method in methods:
                        # One untimed primitive warmup per representation/shape.
                        fn=(lambda:assign_nearest(x,init,chunk_size=cfg.assignment_chunk_size,centroid_chunk_size=cfg.centroid_chunk_size)) if method=='ours_torch' else (lambda:authors_assignment(backend,x,init,cfg.assignment_chunk_size,cfg.centroid_chunk_size))
                        fn();synchronize(device);start=perf_counter();ids,dist=fn();synchronize(device)
                        assigned[method]=dict(wall_seconds=perf_counter()-start,labels=ids.cpu().numpy(),distances=dist.cpu().numpy())
                    a,b=results['ours_torch'],results['authors_native']
                    primitive_a,primitive_b=assigned['ours_torch'],assigned['authors_native']
                    name=f"{case['name']}_seed{seed}_repeat{repeat}"
                    np.savez_compressed(args.output/f'{name}.npz',ours_centroids=a.pop('centroids'),authors_centroids=b.pop('centroids'),init_centroids=ii,
                        ours_assignment=primitive_a['labels'],authors_assignment=primitive_b['labels'],ours_distances=primitive_a['distances'],authors_distances=primitive_b['distances'])
                    arrays=np.load(args.output/f'{name}.npz',allow_pickle=False)
                    row=dict(status='gpu_measured' if gpu else 'cpu_rehearsal',workload=case['name'],seed=seed,repeat=repeat,
                        source_sha256=manifest['source_sha256'],hardware=hardware,data_and_init_sha256=digest,config=asdict(cfg),fits=results,
                        assignment_wall_seconds={m:assigned[m]['wall_seconds'] for m in methods},
                        exact_assignment_labels=bool(np.array_equal(primitive_a['labels'],primitive_b['labels'])),
                        max_abs_distance_difference=float(np.max(np.abs(primitive_a['distances']-primitive_b['distances']))),
                        max_abs_center_difference=float(np.max(np.abs(arrays['ours_centroids']-arrays['authors_centroids']))),
                        attribution='norm/dot numerical and host-control differences; no algorithmic or matched-quality speedup inferred',
                        artifact_sha256=file_hash(args.output/f'{name}.npz'))
                    arrays.close();write_json(args.output/f'{name}.json',row)
                    manifest['runs'].append(f'{name}.json');manifest['record_sha256'][f'{name}.json']=file_hash(args.output/f'{name}.json')
                    write_json(args.output/'manifest.json',manifest)
        if source_hash(ROOT)!=manifest['source_sha256']:raise RuntimeError('source changed during native comparison')
        manifest.update(status='complete' if gpu else 'cpu_rehearsal_complete',completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:manifest.update(status='failed',error=str(error));raise
    finally:write_json(args.output/'manifest.json',manifest)
    write_json(args.output/'summary.json',summarize(args.output,allow_cpu=args.cpu_rehearsal))
    print(manifest['status'],len(manifest['runs']),'paired native backend comparisons')


if __name__=='__main__':main()
