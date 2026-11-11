"""Local protocol/data checks and explicit timer mocks; no GPU verification."""

from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from e_means.gpu_protocol import (array_hash,file_hash,junit_counts,seed_summary,source_hash,
                                  time_to_target,verify_gate)
from e_means.gpu_workloads import load_workload,validate_config

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    c=json.loads((ROOT/"configs/gpu_smoke.json").read_text())
    c["workloads"]=c["workloads"][:1]
    c["workloads"][0].update(n_train=31,n_test=7,n_features=8,n_clusters=3)
    return c


def test_workload_pairing_and_data_seed_independence(config):
    validate_config(config)
    x,test,init,digest=load_workload(config["workloads"][0],0,"float32",ROOT)
    xx,tt,ii,other=load_workload(config["workloads"][0],1,"float32",ROOT)
    assert_array_equal(xx,x);assert_array_equal(tt,test)
    assert not np.array_equal(init,ii)
    assert digest==array_hash(x,test,init) and digest!=other


@pytest.mark.parametrize("section,key,value",[(None,"warmups",0),(None,"repeats",0),(None,"seeds",[1,1]),
    (None,"dtypes",["float16"]),(None,"target_relative_slack",-1),("common","seed",10),
    ("e_means","n_clusters",1),("lloyd","dtype","float64")])
def test_bad_config(config,section,key,value):
    target=config if section is None else config[section]
    target[key]=value
    with pytest.raises(ValueError): validate_config(config)


def test_embedding_pin_and_prefix(tmp_path):
    x=np.arange(80,dtype=np.float32).reshape(20,4);test=x[:7]+1
    path=tmp_path/"vectors.npz";np.savez(path,train=x,test=test)
    case=dict(kind="embedding",path=str(path),array_sha256=array_hash(x,test),n_clusters=3,train_limit=8,test_limit=4)
    a,b,_,_=load_workload(case,0,"float64",ROOT)
    assert a.shape==(8,4) and b.shape==(4,4)
    with pytest.raises(ValueError):load_workload(dict(case,array_sha256="bad"),0,"float32",ROOT)
    with pytest.raises(ValueError):load_workload(dict(case,train_limit=21),0,"float32",ROOT)


def test_target_observations_and_censoring():
    points=[dict(epoch=0,stage="main_state",test_mse=10,wall_seconds=.2),
            dict(epoch=1,stage="main_state",test_mse=8,wall_seconds=.8),
            dict(epoch=2,stage="final_readout",test_mse=6,wall_seconds=1.4)]
    hit=time_to_target(points,7)
    assert hit["seconds"]==1.4 and hit["stage"]=="final_readout"  # no invented interpolation
    miss=time_to_target(points,5)
    assert miss["seconds"] is None and not miss["reached"] and miss["censored_at_seconds"]==1.4
    with pytest.raises(ValueError):time_to_target(points[::-1],0)
    with pytest.raises(ValueError):time_to_target(points[::-1],100)
    assert time_to_target(points,10)["seconds"]==.2
    assert time_to_target([points[-1]],10)["seconds"]==1.4  # returned readout has its own availability


def test_seed_aggregation_does_not_treat_repeats_as_independent():
    summary=seed_summary([dict(seed=0,value=1),dict(seed=0,value=3),dict(seed=1,value=6),dict(seed=1,value=6)],"value")
    assert summary["mean"]==4 and summary["std"]==np.sqrt(8) and summary["independent_seeds"]==2


def test_gate_rejects_skips_and_source_or_device_changes(tmp_path):
    (tmp_path/"pyproject.toml").write_text("test")
    hardware=dict(name="protocol test only, not actual hardware")
    report=dict(status="cuda_verified",source_sha256=source_hash(tmp_path),hardware=hardware,
                suite=dict(tests=200,cuda_cases=28,failures=0,errors=0,skipped=0))
    verify_gate(report,tmp_path,hardware)  # validates metadata policy, not a device probe
    with pytest.raises(ValueError):verify_gate(dict(report,status="prepared"),tmp_path,hardware)
    with pytest.raises(ValueError):verify_gate(dict(report,suite=dict(report["suite"],skipped=1)),tmp_path,hardware)
    with pytest.raises(ValueError):verify_gate(report,tmp_path,dict(name="different"))
    (tmp_path/"pyproject.toml").write_text("changed")
    with pytest.raises(ValueError):verify_gate(report,tmp_path,hardware)


def test_junit_counts_actual_cases_not_collection(tmp_path):
    path=tmp_path/"suite.xml"
    path.write_text('<testsuites><testsuite><testcase classname="tests.test_cuda_validation" name="check[cuda-float32]"/>'
                    '<testcase classname="tests.test_cuda_validation" name="check[cpu-float32]"/>'
                    '<testcase classname="tests.test_cuda_validation" name="check[cuda-float64]"><skipped/></testcase>'
                    '</testsuite></testsuites>')
    assert junit_counts(path)==dict(tests=3,skipped=1,failures=0,errors=0,cuda_cases=2)


def test_prepare_only_and_no_hardware_report(config,tmp_path):
    pytest.importorskip("torch")
    cfg=tmp_path/"config.json";cfg.write_text(json.dumps(config))
    output=tmp_path/"prepared"
    result=subprocess.run([sys.executable,str(ROOT/"scripts/benchmark_cuda.py"),"--config",str(cfg),"--output",str(output),"--prepare-only"],
                          text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    m=json.loads((output/"manifest.json").read_text())
    assert m["status"]=="prepared" and m["gpu_verified"] is False and m["runs"]==[] and m["hardware"] is None
    assert file_hash(output/"source.zip")==m["source_archive_sha256"]
    # Even on NVIDIA, this subprocess deliberately hides all devices. It must
    # not start pytest recursively or create a successful verification report.
    unavailable=tmp_path/"unavailable"
    result=subprocess.run([sys.executable,str(ROOT/"scripts/validate_cuda.py"),"--output",str(unavailable)],
                          env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),text=True,capture_output=True)
    assert result.returncode==2,result.stderr
    assert json.loads((unavailable/"report.json").read_text())["status"]=="hardware_unavailable"
    assert not (unavailable/"suite.xml").exists()


def test_range_integrity_and_prepared_summary_rejected(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    fetch=importlib.import_module("fetch_gpu_embeddings")
    source=json.loads((ROOT/"data/glove100_source.json").read_text())
    with pytest.raises(ValueError,match="range mismatch"):
        fetch.prepare(source,dict(train=b"bad",test=b"bad"),tmp_path/"vectors.npz")
    summary=importlib.import_module("summarize_cuda")
    (tmp_path/"manifest.json").write_text(json.dumps(dict(status="prepared",gpu_verified=False)))
    with pytest.raises(ValueError,match="preparation"):
        summary.summarize(tmp_path)


def test_timer_order_with_explicit_cuda_api_mock(monkeypatch,config):
    """Protocol-only: fake events/memory prove call order, not CUDA numerics."""
    torch=pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    bench=importlib.import_module("benchmark_cuda")
    import e_means.torch_backend as backend
    original=backend.TorchLloydKMeans
    class CPURehearsal(original):
        def __init__(self,cfg,device=None):super().__init__(cfg,device="cpu")
    monkeypatch.setattr(backend,"TorchLloydKMeans",CPURehearsal)
    calls=[]
    class FakeEvent:
        def __init__(self,**kwargs):pass
        def record(self):calls.append("event")
        def elapsed_time(self,other):return 12.0
    monkeypatch.setattr(torch.cuda,"Event",FakeEvent)
    monkeypatch.setattr(torch.cuda,"synchronize",lambda: calls.append("sync"))
    monkeypatch.setattr(torch.cuda,"reset_peak_memory_stats",lambda: calls.append("reset"))
    for name,value in (("memory_allocated",100),("memory_reserved",200),("max_memory_allocated",150),("max_memory_reserved",250)):
        monkeypatch.setattr(torch.cuda,name,lambda value=value:value)
    x,test,init,_=load_workload(config["workloads"][0],0,"float64",ROOT)
    row,arrays=bench.measure(config,config["workloads"][0],"lloyd",0,"float64",*(torch.tensor(a) for a in (x,test,init)))
    assert calls[:3]==["sync","reset","event"] and calls[-2:]==["event","sync"]
    assert row["cuda_event_elapsed_seconds"]==.012 and row["incremental_peak_allocated_bytes"]==50
    assert row["convergence"][-1]["stage"]=="final_readout" and arrays["centroids"].shape==(3,8)
    assert "status" not in row  # this mock cannot issue a gpu_measured/verified certificate
