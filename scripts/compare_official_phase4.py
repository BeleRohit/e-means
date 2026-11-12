"""Pinned authors' Faiss-CPU comparison, isolated from PyTorch/OpenMP.

This is NOT an official GPU kernel comparison. FP32, identical caller-owned
initialization/batch order, retained tails, mass 2**-16, revival disabled.
Disabling revival isolates arithmetic/assignment throughput from donor RNG
differences; paper-default results remain separate and unchanged.
"""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import types
import zipfile
from time import perf_counter

import numpy as np
from e_means.assignment import assign_nearest,batch_statistics
from e_means.state import EMAState,last_epoch_mean
from e_means.schedules import cosine_alpha
from e_means.gpu_workloads import load_workload
from e_means.gpu_protocol import file_hash,source_hash,source_paths,write_json,seed_summary

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.fetch_reference import HASHES,COMMIT


def load_official(root):
    for name,digest in HASHES.items():
        if file_hash(root/name)!=digest:raise ValueError('pinned authors source mismatch: '+name)
    name='_phase4_authors'
    package=types.ModuleType(name);package.__path__=[str(root)];sys.modules[name]=package
    spec=importlib.util.spec_from_file_location(name+'.e_means',root/'e_means.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def fit(method,official,x,init,seed,epochs,batch,alpha):
    if method not in ('numpy_primitives','authors_faiss_cpu'):raise ValueError('unknown method')
    k=len(init);steps=epochs*((len(x)+batch-1)//batch)
    order_rng=np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[1])
    if method=='authors_faiss_cpu':
        model=official.NumpyEmeans(x.shape[1],k,nepochs=epochs,batch_size=batch,alpha0=alpha,
                                  unused_threshold=0,capture_assignments=True,max_points_per_centroid=None)
        model.begin(len(x),init)
    else:state=EMAState.initialize(init,2**-16)
    step=0
    for epoch in range(epochs):
        order=order_rng.permutation(len(x));sums=np.zeros_like(init);counts=np.zeros(k,dtype='float32')
        for offset in range(0,len(x),batch):
            rows=x[order[offset:offset+batch]]
            if method=='authors_faiss_cpu':model.update(model.accumulate(rows))
            else:
                labels,_=assign_nearest(rows,state.centroids(),dtype='float32')
                s,n=batch_statistics(rows,labels,k)
                if epoch==epochs-1:sums+=s;counts+=n
                state.update(s,n,cosine_alpha(step,steps,alpha))
            step+=1
        if method=='authors_faiss_cpu':model.end_epoch()
    return model.finalize() if method=='authors_faiss_cpu' else last_epoch_mean(sums,counts,state.centroids())


def summarize(root):
    """CPU-only artifact audit/aggregation; never interprets these as GPU data."""
    from collections import defaultdict
    m=json.loads((root/'manifest.json').read_text())
    if m['status']!='cpu_complete' or m['gpu_verified']:raise ValueError('complete CPU comparison required')
    if file_hash(root/'config.json')!=m['config_sha256'] or file_hash(root/'source.zip')!=m['source_archive_sha256']:
        raise ValueError('CPU source/config snapshot changed')
    cfg=json.loads((root/'config.json').read_text())
    expected={f"{case['name']}_seed{seed}_{method}_repeat{rep}.json" for case in cfg['workloads']
              for seed in m['seeds'] for method in ('numpy_primitives','authors_faiss_cpu') for rep in range(m['repeats'])}
    if set(m['runs'])!=expected or len(m['runs'])!=len(expected):raise ValueError('incomplete/duplicate CPU comparison')
    groups=defaultdict(list);pairs=defaultdict(dict)
    for name in m['runs']:
        if file_hash(root/name)!=m['record_sha256'][name]:raise ValueError('CPU record changed')
        r=json.loads((root/name).read_text())
        if r['status']!='cpu_measured' or r['source_sha256']!=m['source_sha256']:raise ValueError('mixed CPU/source data')
        if file_hash(root/name.replace('.json','.npz'))!=r['artifact_sha256']:raise ValueError('CPU arrays changed')
        with np.load(root/name.replace('.json','.npz'),allow_pickle=False) as f:
            pairs[(r['workload'],r['seed'],r['repeat'])][r['method']]=dict(centroids=f['centroids'],init=f['init_centroids'],digest=r['data_and_init_sha256'])
        groups[(r['workload'],r['method'])].append(r)
    parity=[]
    for key,pair in pairs.items():
        ours,authors=pair['numpy_primitives'],pair['authors_faiss_cpu']
        if ours['digest']!=authors['digest'] or not np.array_equal(ours['init'],authors['init']):raise ValueError('CPU initialization/input mismatch')
        parity.append(dict(workload=key[0],seed=key[1],repeat=key[2],
            exact_centers=bool(np.array_equal(ours['centroids'],authors['centroids'])),
            max_abs_center_difference=float(np.max(np.abs(ours['centroids']-authors['centroids'])))))
    return dict(status='cpu_measured',gpu_verified=False,verified_fits=len(expected),parity=parity,
                summary=[dict(workload=key[0],method=key[1],**{metric:seed_summary(rs,metric) for metric in ('solver_wall_seconds','total_wall_seconds','train_mse','test_mse')}) for key,rs in sorted(groups.items())],
                caveat='single-host CPU arithmetic/assignment comparison with revival disabled; not official GPU speedup evidence')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/gpu_smoke.json')
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);p.add_argument('--repeats',type=int,default=2)
    args=p.parse_args()
    if args.repeats<1 or not args.seeds or len(set(args.seeds))!=len(args.seeds) or any(s<0 for s in args.seeds):raise ValueError('positive repeats and distinct nonnegative seeds required')
    # No Torch import in this process: preserve the bundled OpenMP isolation.
    official=load_official(args.reference)
    import faiss
    faiss.omp_set_num_threads(1)
    config=json.loads(args.config.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'config.json',config)
    with zipfile.ZipFile(args.output/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths(ROOT):archive.write(path,path.relative_to(ROOT))
    import importlib.metadata
    try:scipy=importlib.metadata.version('scipy')
    except importlib.metadata.PackageNotFoundError:scipy=None
    manifest=dict(status='running',gpu_verified=False,scope='local CPU assignment/arithmetic comparison; not GPU speedup evidence',
        source_sha256=source_hash(ROOT),reference_commit=COMMIT,reference_hashes=HASHES,
        config_sha256=file_hash(args.output/'config.json'),source_archive_sha256=file_hash(args.output/'source.zip'),seeds=args.seeds,repeats=args.repeats,
        environment=dict(platform=platform.platform(),python=sys.version,numpy=np.__version__,faiss=faiss.__version__,
                         faiss_threads=faiss.omp_get_max_threads(),scipy=scipy,omp=os.getenv('OMP_NUM_THREADS'),openblas=os.getenv('OPENBLAS_NUM_THREADS')),
        differences=dict(precision='FP32 storage/statistics; common FP64 evaluator',mass=2**-16,revival_threshold=0,
            tail='keep',batch_order='independent Generator SeedSequence(seed).spawn(3)[1]',
            own_driver='existing NumPy assignment/state primitives; no epoch metrics inside solver',
            reference_driver='unmodified caller-driven NumpyEmeans accumulate/update/end_epoch/finalize',
            backend='ours direct differences; authors Faiss CPU exact assignment; accumulation/centroid representations differ',
            scipy='authors optional CSR statistics path used when installed; package version recorded'),runs=[],record_sha256={})
    write_json(args.output/'manifest.json',manifest)
    try:
        for case in config['workloads']:
            for seed in args.seeds:
                x,test,init,digest=load_workload(case,seed,'float32',ROOT)
                options=dict(seed=seed,epochs=config['e_means']['epochs'],batch=config['e_means']['batch_size'],alpha=config['e_means']['alpha0'])
                for method in ('numpy_primitives','authors_faiss_cpu'):fit(method,official,x,init,**options)
                for repeat in range(args.repeats):
                    methods=('numpy_primitives','authors_faiss_cpu')
                    if (repeat+seed)%2:methods=methods[::-1]
                    outputs={}
                    for method in methods:
                        start=perf_counter();centers=fit(method,official,x,init,**options);solver=perf_counter()-start
                        eval_start=perf_counter()
                        train_loss=assign_nearest(x,centers,dtype='float64')[1].mean()
                        test_loss=assign_nearest(test,centers,dtype='float64')[1].mean()
                        evaluation=perf_counter()-eval_start
                        name=f"{case['name']}_seed{seed}_{method}_repeat{repeat}"
                        np.savez_compressed(args.output/f'{name}.npz',centroids=centers,init_centroids=init)
                        record=dict(status='cpu_measured',workload=case['name'],seed=seed,repeat=repeat,method=method,
                            train_mse=float(train_loss),test_mse=float(test_loss),solver_wall_seconds=solver,
                            common_fp64_evaluation_seconds=evaluation,total_wall_seconds=solver+evaluation,
                            data_and_init_sha256=digest,source_sha256=manifest['source_sha256'],settings=options,
                            artifact_sha256=file_hash(args.output/f'{name}.npz'))
                        write_json(args.output/f'{name}.json',record);manifest['runs'].append(f'{name}.json');outputs[method]=centers
                        manifest['record_sha256'][f'{name}.json']=file_hash(args.output/f'{name}.json')
                    # A validation result, not assumed algorithmic equivalence.
                    parity=dict(workload=case['name'],seed=seed,repeat=repeat,
                        max_abs_center_difference=float(np.max(np.abs(outputs[methods[0]]-outputs[methods[1]]))),
                        allclose_atol_3e_6_rtol_1e_5=bool(np.allclose(outputs[methods[0]],outputs[methods[1]],atol=3e-6,rtol=1e-5)))
                    write_json(args.output/f"{case['name']}_seed{seed}_repeat{repeat}_parity.json",parity)
                write_json(args.output/'manifest.json',manifest)
        if source_hash(ROOT)!=manifest['source_sha256']:raise RuntimeError('source changed during CPU comparison')
        manifest.update(status='cpu_complete',completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:manifest.update(status='failed',error=str(error));raise
    finally:write_json(args.output/'manifest.json',manifest)
    write_json(args.output/'summary.json',summarize(args.output))
    print(len(manifest['runs']),'actual CPU fits; no GPU comparison claimed')


if __name__=='__main__':main()
