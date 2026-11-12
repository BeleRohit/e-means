"""Benchmark-only observation of existing algorithms; no kernel changes.

Process-local patching is deliberately serial and restored on failure.
CUDA events are collected without adding per-range synchronization. Nested
ranges overlap and MUST NOT be summed into a disjoint time breakdown.
"""
from collections import defaultdict
from contextlib import contextmanager, ExitStack
from functools import wraps
from time import perf_counter
from unittest.mock import patch

import torch
from . import torch_backend as backend


class Observation:
    def __init__(self, *, events=False, ranges=False):
        self.events=events
        self.ranges=ranges
        self.scope='training'
        self.calls=defaultdict(int)
        self.host_seconds=defaultdict(float)
        self.pairs=defaultdict(int)
        self.samples=[]
        self.points=[]

    @contextmanager
    def span(self,name):
        start=end=None
        if self.events:
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record()
        origin=perf_counter()
        with (torch.profiler.record_function('emeans::'+name) if self.ranges else ExitStack()):
            try: yield
            finally:
                self.calls[name]+=1
                self.host_seconds[name]+=perf_counter()-origin
                if self.events:
                    end.record();self.samples.append((name,start,end))

    def report(self):
        if self.events: torch.cuda.synchronize()
        times=defaultdict(float)
        for name,start,end in self.samples: times[name]+=start.elapsed_time(end)/1000
        return dict(ranges={name:dict(calls=self.calls[name],host_inclusive_seconds=self.host_seconds[name],
                                     cuda_stream_span_seconds=times[name] if self.events else None)
                            for name in sorted(self.calls)}, distance_pairs=dict(self.pairs),
                    timing_caveat='inclusive nested ranges overlap; CUDA stream spans contain submission gaps; do not sum or call kernel time')

    @contextmanager
    def instrument(self, model, *, full=True):
        """Wrap functions, not algorithm loops; outputs/RNG streams preserved."""
        observation=self
        with ExitStack() as stack:
            def wrap(obj,name,label):
                original=getattr(obj,name)
                @wraps(original)
                def call(*args,**kwargs):
                    with observation.span(label): return original(*args,**kwargs)
                stack.enter_context(patch.object(obj,name,call))
            components=[(backend,'batch_statistics','statistics'),
                                   (backend,'last_epoch_mean','centroid_readout'),
                                   (backend.TorchEMAState,'update','ema_update'),
                                   (backend.TorchEMAState,'revive','revival'),
                                   (backend.TorchEMAState,'split','split')] if full else []
            for obj,name,label in components:
                wrap(obj,name,label)
            # Last-epoch means are also used for state readouts: the top-level
            # finalizer is identified by caller context, not by that primitive.
            assignment=model._assignment
            def assigned(x,c):
                observation.pairs[observation.scope]+=len(x)*len(c)
                with observation.span('assignment/'+observation.scope): return assignment(x,c)
            stack.enter_context(patch.object(model,'_assignment',assigned))
            for name,scope in [('_record','evaluation'),('_publish','publish')]:
                original=getattr(model,name)
                def scoped(*args,_original=original,_scope=scope,**kwargs):
                    previous=observation.scope;observation.scope=_scope
                    try:
                        with observation.span(_scope): result=_original(*args,**kwargs)
                        if _scope=='evaluation':
                            observation.points.append(dict(epoch=result.epoch,train_mse=result.train_mse,test_mse=result.test_mse,
                                wall_seconds=perf_counter()-observation.origin,training_distance_pairs=observation.pairs['training'],
                                all_distance_pairs=sum(observation.pairs.values()),stage='main_state'))
                        return result
                    finally: observation.scope=previous
                stack.enter_context(patch.object(model,name,scoped))
            if full:
                wrap(model,'_inputs','input_validation')
                wrap(model,'_sync','explicit_sync')
            self.origin=perf_counter()
            yield self
