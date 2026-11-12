"""Independent Phase 4 design, work accounting and observer correctness."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from e_means import EMeansConfig,LloydConfig
from e_means.quality_protocol import grid_target,recipe_config,variants,distance_work
from e_means.schedules import paper_batch_size,paper_alpha0


def test_laws_are_resolved_for_each_actual_horizon():
    base=EMeansConfig(64,batch_size=256,alpha0=.2)
    for budget in [1,2,3,5]:
        cfg,recipe=recipe_config(base,8192,budget,variants()[1])
        b=paper_batch_size(8192,64,budget)
        assert recipe['batch_size']==b
        assert recipe['alpha0']==paper_alpha0(8192,64,budget,b)
        assert cfg.alpha0 is None and cfg.batch_size is None
        assert recipe['alpha_source']=='paper_table_2'


def test_sensitivity_couples_batch_law_and_step_law():
    base=EMeansConfig(64)
    _,automatic=recipe_config(base,8192,3,variants()[1])
    for variant in variants()[2:]:
        _,recipe=recipe_config(base,8192,3,variant)
        if 'batch_factor' in variant:
            assert recipe['batch_size']==automatic['batch_size']*variant['batch_factor']
            assert recipe['alpha0']==paper_alpha0(8192,64,3,recipe['batch_size'])
        else:
            assert recipe['batch_size']==automatic['batch_size']
            assert recipe['alpha0']==min(1,automatic['alpha0']*variant['alpha_factor'])


def test_returned_budget_grid_is_not_fastest_repeat_or_main_ema():
    rows=[dict(budget=1,train_mse=8,test_mse=10,wall_seconds=.7),
          dict(budget=2,train_mse=6,test_mse=7,wall_seconds=1.2),
          dict(budget=5,train_mse=5,test_mse=6,wall_seconds=1.)]
    targets=dict(train_mse=7,test_mse=8)
    hit=grid_target(rows,targets,'joint')
    assert hit['budget']==2 and hit['seconds']==1.2
    assert hit['grid_search_wall_seconds']==pytest.approx(2.9)
    miss=grid_target(rows,dict(train_mse=4,test_mse=4),'joint')
    assert miss['seconds'] is None and miss['max_observed_budget']==5 and not miss['reached']
    # Train/test are separate; train-only hit cannot stand in for joint/test.
    assert grid_target(rows,targets,'train_mse')['budget']==2
    with pytest.raises(ValueError):grid_target(rows+rows[:1],targets,'test_mse')


@pytest.mark.parametrize('bad',[-1,float('nan'),float('inf')])
def test_bad_target(bad):
    with pytest.raises(ValueError):grid_target([dict(budget=1,train_mse=1,test_mse=1,wall_seconds=1)],dict(train_mse=bad,test_mse=1),'joint')


@pytest.mark.parametrize('method',['lloyd','emeans','slow_ema'])
@pytest.mark.parametrize('evaluation',['every_epoch','final_only'])
def test_diagnostic_replay_counts_and_final_only_parity(method,evaluation):
    torch=pytest.importorskip('torch')
    from e_means.quality_experiments import timed_fit,observed_fit
    torch.set_num_threads(1)
    x=torch.tensor(np.random.default_rng(1).normal(size=(17,3)),dtype=torch.float64)
    test=x[:5]+.2;init=x[:3].clone()
    cfg=(LloydConfig(3,max_iter=3) if method=='lloyd' else
         EMeansConfig(3,epochs=3,batch_size=4,alpha0=.2,drop_last=True,
                      finalize='slow_ema' if method=='slow_ema' else 'last_epoch_mean'))
    row,arrays=timed_fit(cfg,method,x,test,init,'cpu',evaluation)
    other,other_arrays=timed_fit(cfg,method,x,test,init,'cpu','final_only' if evaluation=='every_epoch' else 'every_epoch')
    np.testing.assert_array_equal(arrays['centroids'],other_arrays['centroids'])
    diagnostic=observed_fit(cfg,method,x,test,init,'cpu',evaluation,arrays)
    pairs=diagnostic['distance_pairs']
    assert pairs['training']==distance_work(17,3,row['actual_epochs'],batch=None if method=='lloyd' else 4,drop_last=method!='lloyd')
    assert pairs['publish']==17*3 and pairs['final_test']==5*3
    if evaluation=='every_epoch': assert pairs['evaluation']==(17+5)*3*(row['actual_epochs']+1)
    else: assert pairs.get('evaluation',0)==0
    assert diagnostic['replay_outputs_exact'] and diagnostic['profiling_enabled'] is False


def test_observer_restores_functions_after_exception():
    pytest.importorskip('torch')
    from e_means import torch_backend as backend
    from e_means.profiling import Observation
    model=backend.TorchEMeans(EMeansConfig(2),device='cpu')
    original=backend.TorchEMAState.update
    with pytest.raises(RuntimeError):
        with Observation().instrument(model):
            assert backend.TorchEMAState.update is not original
            raise RuntimeError('fixture failure')
    assert backend.TorchEMAState.update is original


def test_cpu_profiler_exports_actual_trace(tmp_path):
    torch=pytest.importorskip('torch')
    from e_means.quality_experiments import timed_fit,observed_fit
    x=torch.arange(24,dtype=torch.float64).reshape(8,3)
    cfg=EMeansConfig(2,epochs=1,batch_size=4,alpha0=.2)
    _,arrays=timed_fit(cfg,'emeans',x,x[:2],x[:2],'cpu','every_epoch')
    trace=tmp_path/'trace.json'
    result=observed_fit(cfg,'emeans',x,x[:2],x[:2],'cpu','every_epoch',arrays,profile=True,trace_path=trace)
    assert trace.exists() and result['cuda_kernel_events']==0
    assert result['operators'] and result['ranges']['ema_update']['calls']==2
    assert result['ranges']['finalization_replay']['calls']==1


def test_pilot_matrix_preserves_identical_init_and_has_no_duplicate_jobs():
    from scripts.phase4_cuda import load_config,jobs,job_id
    config,base=load_config(Path(__file__).resolve().parents[1]/'configs/phase4_pilot.json')
    planned=list(jobs(config,base,{'glove100':16384}))
    identities=[j['case']['name']+job_id(j) for j in planned]
    assert len(set(identities))==len(identities)
    assert all(j['config']['seed']==j['seed'] for j in planned)
    assert len(planned)==312


@pytest.fixture
def cpu_phase4_run(tmp_path):
    import subprocess,sys
    from scripts.phase4_cuda import ROOT
    baseline=json.loads((ROOT/'configs/gpu_smoke.json').read_text())
    baseline['workloads']=baseline['workloads'][:1]
    baseline['workloads'][0].update(n_train=17,n_test=5,n_features=3,n_clusters=2)
    base_path=tmp_path/'base.json';base_path.write_text(json.dumps(baseline))
    config=json.loads((ROOT/'configs/phase4_pilot.json').read_text())
    config.update(baseline_config=str(base_path),workloads=[baseline['workloads'][0]['name']],seeds=[0],repeats=1,
                  lloyd_budgets=[1],emeans_budgets=[1],sensitivity_budget=1,profile_budget=1)
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    output=tmp_path/'run'
    result=subprocess.run([sys.executable,str(ROOT/'scripts/phase4_cuda.py'),'--config',str(path),
                           '--output',str(output),'--mode','benchmark','--cpu-rehearsal'],capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    return output


def test_cpu_pipeline_cannot_masquerade_as_cuda_and_preserves_censoring(cpu_phase4_run):
    from scripts.summarize_phase4 import summarize
    root=cpu_phase4_run
    with pytest.raises(ValueError):summarize(root)
    result=summarize(root,allow_cpu=True)
    assert result['status']=='cpu_rehearsal' and result['verified_runs']==18
    assert result['matched_grid'] and any(not r['reached'] for r in result['matched_grid'])
    assert all(r['paired_lloyd_over_contender_seconds'] is None for r in result['paired_comparisons']
               if not(r['lloyd_reached'] and r['contender_reached']))
    manifest=json.loads((root/'manifest.json').read_text())
    path=root/manifest['runs'][0]
    data=json.loads(path.read_text());data['train_mse']=0;path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='JSON changed'):summarize(root,allow_cpu=True)


@pytest.mark.cuda
def test_actual_cuda_profiler_and_event_probe(tmp_path):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('actual NVIDIA hardware required')
    from e_means.gpu_protocol import configure_torch
    from e_means.quality_experiments import timed_fit,observed_fit
    configure_torch()
    x=torch.arange(24,device='cuda',dtype=torch.float64).reshape(8,3)
    cfg=EMeansConfig(2,epochs=1,batch_size=4,alpha0=.2)
    row,arrays=timed_fit(cfg,'emeans',x,x[:2],x[:2],'cuda','every_epoch')
    result=observed_fit(cfg,'emeans',x,x[:2],x[:2],'cuda','every_epoch',arrays,profile=True,trace_path=tmp_path/'cuda.json')
    assert row['cuda_event_seconds']>0 and result['cuda_kernel_events']>0
    assert result['ranges']['ema_update']['cuda_stream_span_seconds']>0


def test_official_source_pin_checked_before_import(tmp_path):
    from scripts.compare_official_phase4 import load_official
    (tmp_path/'base.py').write_text('unverified authors source')
    with pytest.raises(ValueError,match='pinned authors source mismatch'):load_official(tmp_path)


@pytest.mark.reference
def test_official_comparison_cli_isolated_and_audited(tmp_path):
    import importlib.util,os,subprocess,sys
    reference=os.getenv('EMEANS_REFERENCE_DIR')
    if reference is None or importlib.util.find_spec('faiss') is None:pytest.skip('pinned Faiss reference required')
    from scripts.phase4_cuda import ROOT
    baseline=json.loads((ROOT/'configs/gpu_smoke.json').read_text())
    baseline['workloads']=baseline['workloads'][:1]
    baseline['workloads'][0].update(n_train=17,n_test=5,n_features=3,n_clusters=2)
    config=tmp_path/'config.json';config.write_text(json.dumps(baseline))
    output=tmp_path/'official'
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    r=subprocess.run([sys.executable,str(ROOT/'scripts/compare_official_phase4.py'),'--reference',reference,
                      '--config',str(config),'--output',str(output),'--seeds','0','--repeats','1'],env=env,capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stdout+r.stderr
    from scripts.compare_official_phase4 import summarize
    summary=summarize(output)
    assert summary['status']=='cpu_measured' and not summary['gpu_verified'] and summary['verified_fits']==2
    assert summary['parity'][0]['max_abs_center_difference']<3e-6
    manifest=json.loads((output/'manifest.json').read_text());path=output/manifest['runs'][0]
    r=json.loads(path.read_text());r['solver_wall_seconds']=0;path.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='record changed'):summarize(output)


@pytest.mark.reference
def test_native_author_adapter_cpu_compatibility(tmp_path):
    import os,subprocess,sys
    reference=os.getenv('EMEANS_REFERENCE_DIR')
    if reference is None or not (Path(reference)/'torch_backend.py').exists():pytest.skip('optional pinned native authors files required')
    from scripts.phase4_cuda import ROOT
    baseline=json.loads((ROOT/'configs/gpu_smoke.json').read_text())
    baseline['workloads']=baseline['workloads'][:1]
    baseline['workloads'][0].update(n_train=19,n_test=5,n_features=4,n_clusters=2)
    base=tmp_path/'baseline.json';base.write_text(json.dumps(baseline))
    config=json.loads((ROOT/'configs/phase4_pilot.json').read_text())
    config.update(baseline_config=str(base),workloads=[baseline['workloads'][0]['name']],seeds=[0],repeats=1,sensitivity_budget=2)
    path=tmp_path/'native.json';path.write_text(json.dumps(config));output=tmp_path/'native'
    r=subprocess.run([sys.executable,str(ROOT/'scripts/compare_native_phase4.py'),'--config',str(path),'--reference',reference,
                      '--output',str(output),'--cpu-rehearsal'],capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stdout+r.stderr
    from scripts.compare_native_phase4 import summarize
    with pytest.raises(ValueError):summarize(output)
    result=summarize(output,allow_cpu=True)
    assert result['paired_records']==1 and result['status']=='cpu_rehearsal'
    assert result['parity'][0]['exact_assignment_labels']
    assert result['parity'][0]['max_abs_center_difference']<3e-6


def test_deadline_guard_only_deletes_saved_id_and_verifies_readback():
    from types import SimpleNamespace
    from runpod.watchdog import terminate
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        if len(calls)==1:return SimpleNamespace(returncode=0,stdout='{"id":"savedpod123"}',stderr='')
        if len(calls)==2:return SimpleNamespace(returncode=0,stdout='{"deleted":true}',stderr='')
        return SimpleNamespace(returncode=1,stdout='',stderr='{"code":"not_found"}')
    assert terminate('savedpod123',run)
    assert [c[-2:] for c in calls]==[['get','savedpod123'],['delete','savedpod123'],['get','savedpod123']]
    with pytest.raises(ValueError):terminate('invalid; shell',run)
