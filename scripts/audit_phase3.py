"""Re-audit immutable A40 exports and quantify observed Phase 3 timing scope."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.summarize_cuda import summarize
from e_means.gpu_protocol import file_hash,seed_summary,write_json


def audit(baseline):
    groups=defaultdict(list);exports=[]
    for study in ('a40-smoke-002','a40-research-001'):
        archive=baseline/(study+'.tar.gz')
        if file_hash(archive)!=(baseline/(study+'.tar.gz.sha256')).read_text().split()[0]:raise ValueError('export checksum mismatch')
        run=baseline/study/'benchmark';checked=summarize(run)
        manifest=json.loads((run/'manifest.json').read_text())
        exports.append(dict(study=study,verified_runs=checked['verified_runs'],archive_sha256=file_hash(archive),
                            source_sha256=manifest['source_sha256'],hardware=manifest['hardware']))
        if study!='a40-research-001':continue
        for name in manifest['runs']:
            r=json.loads((run/name).read_text())
            r['actual_epochs']=r['history'][-1]['epoch']
            r['unattributed_wall_seconds']=r['resident_fit_wall_seconds']-r['training_loop_seconds']-r['evaluation_seconds']-r['finalization_seconds']
            groups[(r['workload'],r['dtype'],r['method'])].append(r)
    rows=[]
    for key,records in sorted(groups.items()):
        rows.append(dict(zip(('workload','dtype','method'),key),
            epoch_range=[min(r['actual_epochs'] for r in records),max(r['actual_epochs'] for r in records)],
            updates_range=[min(r['history'][-1]['updates'] for r in records),max(r['history'][-1]['updates'] for r in records)],
            **{m:seed_summary(records,m) for m in ('resident_fit_wall_seconds','training_loop_seconds','evaluation_seconds','finalization_seconds','unattributed_wall_seconds','train_mse','test_mse')}))
    return dict(status='measured_phase3_audit',verified_runs=sum(e['verified_runs'] for e in exports),exports=exports,rows=rows,
        inference_limits=['Existing training/evaluation timers are observed broad wall intervals, not kernel/host decomposition.',
                          'Unattributed interval includes input validation, publishing and final test evaluation; do not call it host overhead.',
                          'Lloyd stopped at an exact fixed point on Gaussian cases; max_iter=25 was not 25 executed epochs.',
                          'All fixed-budget speed ratios compare unequal final quality; they are not matched-MSE speedups.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=ROOT/'results/runpod/a40-20261111')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--plots',action='store_true')
    args=p.parse_args();result=audit(args.baseline);args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'audit.json',result)
    with (args.output/'timing.csv').open('w',newline='') as stream:
        keys=['workload','dtype','method','min_epochs','max_epochs','fit_seconds','loop_seconds','evaluation_seconds','unattributed_seconds']
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for r in result['rows']:
            writer.writerow(dict(workload=r['workload'],dtype=r['dtype'],method=r['method'],min_epochs=r['epoch_range'][0],max_epochs=r['epoch_range'][1],
                fit_seconds=r['resident_fit_wall_seconds']['mean'],loop_seconds=r['training_loop_seconds']['mean'],
                evaluation_seconds=r['evaluation_seconds']['mean'],unattributed_seconds=r['unattributed_wall_seconds']['mean']))
    if args.plots:
        import os
        os.environ.setdefault('MPLCONFIGDIR',str(args.output/'matplotlib-cache'))
        os.environ.setdefault('XDG_CACHE_HOME',str(args.output/'cache'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        rows=[r for r in result['rows'] if r['dtype']=='float32']
        fig,ax=plt.subplots(figsize=(12,5));bottom=np.zeros(len(rows))
        for metric,label in [('training_loop_seconds','Training-loop wall'),('evaluation_seconds','Epoch-evaluation wall'),('finalization_seconds','Finalizer wall'),('unattributed_wall_seconds','Other wall (not host attribution)')]:
            values=np.array([r[metric]['mean'] for r in rows]);ax.bar(np.arange(len(rows)),values,bottom=bottom,label=label);bottom+=values
        ax.set_xticks(np.arange(len(rows)),[r['workload']+'\n'+r['method'] for r in rows],fontsize=8)
        ax.set_ylabel('Mean resident fit seconds');ax.legend(fontsize=8);ax.set_title('Measured Phase 3 A40 FP32 timing scope; fixed budgets, unequal final MSE')
        fig.tight_layout();fig.savefig(args.output/'phase3_wall_scope.png',dpi=160);plt.close(fig)
    print('Reverified',result['verified_runs'],'original A40 measurements; no new GPU execution')


if __name__=='__main__':main()
