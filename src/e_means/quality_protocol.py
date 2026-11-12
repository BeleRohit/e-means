"""Phase 4 experimental design, independent of tensor kernels.

Returned finalizers are evaluated by fresh fits at each declared horizon.
Changing T changes the cosine schedule and automatic recipe; a prefix of a
long fit is therefore not the output of the short-budget estimator.
"""
from dataclasses import replace
import math

from .config import EMeansConfig, LloydConfig, positive_int
from .schedules import paper_batch_size, paper_alpha0, resolve_recipe


def variants():
    """One-factor sensitivity about Table 2, not a retrospective best sweep."""
    return [dict(name='fixed', batch=256, alpha=.2),
            dict(name='automatic', batch=None, alpha=None),
            dict(name='batch_half', batch_factor=.5),
            dict(name='batch_double', batch_factor=2),
            dict(name='alpha_half', alpha_factor=.5),
            dict(name='alpha_double', alpha_factor=2)]


def recipe_config(base: EMeansConfig, n: int, budget: int, variant: dict):
    positive_int('budget', budget)
    if variant.get('name') not in {v['name'] for v in variants()}:raise ValueError('unknown recipe variant')
    if variant['name'] == 'fixed':
        result = replace(base, epochs=budget, batch_size=variant['batch'], alpha0=variant['alpha'])
    elif variant['name']=='automatic':
        result = replace(base,epochs=budget,batch_size=None,alpha0=None)
    else:
        batch = min(n, max(1, int(paper_batch_size(n, base.n_clusters, budget)*variant.get('batch_factor',1))))
        alpha = min(1., paper_alpha0(n, base.n_clusters, budget, batch)*variant.get('alpha_factor',1))
        result = replace(base, epochs=budget,
                         batch_size=batch if 'batch_factor' in variant else None,
                         alpha0=alpha if 'alpha_factor' in variant else None)
    recipe = resolve_recipe(result,n)
    recipe.update(variant=variant['name'], statistical_scope='small-workload law evaluation, not fit/extrapolation validation',
                  sensitivity_alpha_clipped=variant.get('alpha_factor',1)*paper_alpha0(n,base.n_clusters,budget,recipe['batch_size'])>1,
                  batch_factor=variant.get('batch_factor',1),alpha_factor=variant.get('alpha_factor',1))
    return result, recipe


def grid_target(rows: list[dict], targets: dict, metric: str):
    """First ascending-horizon *returned* solution meeting a fixed target.

    This is observed budget-grid time to quality, not continuous stopping
    latency. Search cost is reported separately. Misses are budget-censored;
    independent wall times need not increase with horizon.
    """
    if metric not in ('train_mse','test_mse','joint') or not rows:
        raise ValueError('nonempty budget grid and valid target metric required')
    for key in ('train_mse','test_mse'):
        if not math.isfinite(targets[key]) or targets[key] < 0:
            raise ValueError('finite nonnegative targets required')
    ordered=sorted(rows,key=lambda r:r['budget'])
    if len({r['budget'] for r in rows}) != len(rows):
        raise ValueError('duplicate horizon in a seed/repeat/method grid')
    for r in ordered:
        positive_int('budget',r['budget'])
        if any(not math.isfinite(r[k]) or r[k]<0 for k in ('train_mse','test_mse','wall_seconds')):
            raise ValueError('invalid measured result')
        if r['wall_seconds']==0:raise ValueError('returned fit latency must be positive')
    def meets(row):
        return all(row[k] <= targets[k] for k in (('train_mse','test_mse') if metric=='joint' else (metric,)))
    hit=next((r for r in ordered if meets(r)),None)
    result=dict(reached=hit is not None, target=targets if metric=='joint' else targets[metric],
                seconds=hit['wall_seconds'] if hit else None, budget=hit['budget'] if hit else None,
                max_observed_budget=ordered[-1]['budget'], grid_search_wall_seconds=sum(r['wall_seconds'] for r in rows),
                censoring=None if hit else 'no returned solution reached target on the declared budget grid',
                interpretation='first ascending-horizon returned solution, no interpolation; not online early stopping')
    return result


def distance_work(n: int, k: int, epochs: int, *, batch: int | None=None, drop_last=False):
    """Exact all-centroid *training* pair count for a declared pass budget.

    Instrumentation separately counts every actual assignment, including
    metric/publish/readout work. A pair entails d coordinate differences.
    """
    for name,v in [('n',n),('k',k),('epochs',epochs)]: positive_int(name,v)
    if batch is not None:positive_int('batch',batch)
    rows=n if batch is None or not drop_last else n//batch*batch
    return rows*k*epochs
