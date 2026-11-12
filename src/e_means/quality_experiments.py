"""Phase 4 fit/trace adapters. Primary timings never enable profiling hooks."""
from dataclasses import asdict
from time import perf_counter
import gc

import numpy as np
import torch

from .estimators import EpochRecord
from .profiling import Observation
from .torch_backend import TorchEMeans, TorchLloydKMeans, last_epoch_mean


def make_model(config, method, device, evaluation):
    if method not in ('lloyd','emeans','slow_ema') or evaluation not in ('every_epoch','final_only'):
        raise ValueError('unknown method/evaluation policy')
    base=TorchLloydKMeans if method=='lloyd' else TorchEMeans
    if evaluation=='every_epoch': return base(config,device=device)
    class FinalOnly(base):
        # Experimental observer policy only. No optimizer update is changed.
        def _record(self,x,c,test,epoch,moving,train_seconds,eval_seconds,updates,revivals):
            return EpochRecord(epoch,None,None,moving,train_seconds,0.,updates,revivals)
    return FinalOnly(config,device=device)


def synchronize(device):
    if torch.device(device).type=='cuda': torch.cuda.synchronize(device)


def returned_metrics(model,test):
    return dict(train_mse=model.mse_,test_mse=model._assignment(test,model.cluster_centers_)[1].double().mean().item())


def timed_fit(config,method,x,test,init,device,evaluation):
    """No profiler/events/counters inside the primary latency interval."""
    model=make_model(config,method,device,evaluation)
    gc.collect();synchronize(device)
    cuda=torch.device(device).type=='cuda'
    if cuda:
        allocated=torch.cuda.memory_allocated(device);reserved=torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        begin,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
    start=perf_counter()
    if cuda: begin.record()
    model.fit(x,init_centroids=init,eval_X=test)
    metrics=returned_metrics(model,test)
    if cuda: end.record()
    synchronize(device)
    elapsed=perf_counter()-start
    record=dict(**metrics,wall_seconds=elapsed,actual_epochs=model.n_iter_,config=asdict(config),
                evaluation=evaluation,method=method,recipe=getattr(model,'recipe_',None),
                integrated_finalizer_wall_seconds=getattr(model,'finalization_seconds_',None),
                history=[asdict(r) for r in model.history_],
                timing='unprofiled synchronized resident fit plus returned train/test evaluation; H2D excluded')
    if cuda:
        record.update(cuda_event_seconds=begin.elapsed_time(end)/1000,
                      baseline_allocated_bytes=allocated,baseline_reserved_bytes=reserved,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    arrays=dict(centroids=model.cluster_centers_.cpu().numpy(),labels=model.labels_.cpu().numpy())
    del model
    return record,arrays


def observed_fit(config,method,x,test,init,device,evaluation,expected,*,profile=False,trace_path=None):
    """Independent diagnostic replay; never substitute this latency for primary."""
    model=make_model(config,method,device,evaluation)
    observer=Observation(events=profile and torch.device(device).type=='cuda',ranges=profile)
    activities=[torch.profiler.ProfilerActivity.CPU]
    if profile and torch.device(device).type=='cuda': activities.append(torch.profiler.ProfilerActivity.CUDA)
    if profile and torch.device(device).type=='cuda' and torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise RuntimeError('CUDA profiler activity unavailable; cannot establish GPU kernel attribution')
    from contextlib import nullcontext
    profiler=(torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False,with_stack=False)
              if profile else nullcontext())
    synchronize(device)
    start=perf_counter()
    with profiler as captured:
        with observer.instrument(model,full=profile):
            model.fit(x,init_centroids=init,eval_X=test)
            observer.scope='final_test'
            metrics=returned_metrics(model,test)
            observer.points.append(dict(**metrics,stage='returned',epoch=model.n_iter_,
                wall_seconds=perf_counter()-start,training_distance_pairs=observer.pairs['training'],
                all_distance_pairs=sum(observer.pairs.values())))
            if profile and method!='lloyd':
                # Replay the exact final readout on already-produced state.
                # Separate micro-observation; not added to fit latency or work.
                with observer.span('finalization_replay'):
                    if config.finalize=='last_epoch_mean':
                        last_epoch_mean(model.last_epoch_sums_,model.last_epoch_counts_,model.ema_centers_)
                    else: model.slow_state_.centroids()
    synchronize(device)
    if not np.array_equal(model.cluster_centers_.cpu().numpy(),expected['centroids']) or not np.array_equal(model.labels_.cpu().numpy(),expected['labels']):
        raise RuntimeError('diagnostic replay changed solver outputs; do not attribute traced timings')
    result=observer.report()
    result.update(diagnostic_wall_seconds=perf_counter()-start,points=observer.points,
                  replay_outputs_exact=True,profiling_enabled=profile,
                  purpose='diagnostic only; primary latency is the separate unprofiled fit')
    if profile:
        if trace_path is None: raise ValueError('trace path required')
        captured.export_chrome_trace(str(trace_path))
        operators=[]
        for event in captured.key_averages():
            operators.append(dict(name=event.key,calls=event.count,self_cpu_us=event.self_cpu_time_total,
                self_device_us=getattr(event,'self_device_time_total',getattr(event,'self_cuda_time_total',0)),
                cpu_inclusive_us=event.cpu_time_total))
        result['operators']=sorted(operators,key=lambda e:e['self_cpu_us'],reverse=True)
        result['cuda_kernel_events']=sum(str(e.device_type).endswith('CUDA') for e in captured.events())
        if torch.device(device).type=='cuda' and not result['cuda_kernel_events']:
            raise RuntimeError('profiler captured zero CUDA events; do not claim GPU kernel attribution')
        sync_names=('synchronize','_local_scalar_dense','aten::item','aten::is_nonzero','Memcpy DtoH')
        result['synchronization_candidates']=[e for e in operators if any(s in e['name'] for s in sync_names)]
        result['unclassified_host_warning']='CPU self time is profiler-attributed, not a disjoint host-minus-GPU residual; inspect timeline for idle gaps and overlap'
    del model
    return result
