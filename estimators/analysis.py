"""Pre-specified paired scene bootstrap and fixed-threshold analyses."""
import argparse
from collections import Counter, defaultdict

import numpy as np

from dev_pipeline import HERE, dump, sha
from experiment import CONDITIONS, load
from gate import METHODS, TARGET, actions

GATES = ('single','dual')
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20261123


def is_move(vector):
    return max(map(abs,vector)) > .25


def attach(rows, frozen):
    for r in rows:
        vectors,scores,reasons,attribution,statuses = actions(r,frozen)
        r.update(actions=vectors,confidence=scores,reasons=reasons,attribution=attribution,legacy_status=statuses)
    return rows


def action_counts(rows, method):
    accounting = []
    for r in rows:
        v = r['actions'][method]
        moved = is_move(v)
        need = r['needs_correction']
        correct = need and moved and np.max(np.abs(np.asarray(v)+r['hr_added'])) <= .25
        executable_correct = bool(r['candidate_correct'])
        accounting.append([need,not need,correct,moved and not need,moved and need and not correct,
                           need and not moved,not moved,executable_correct,
                           correct and executable_correct])
    return np.asarray(accounting,dtype='int64')


COUNT_NAMES = ('move_n','hold_n','correct_n','false_n','wrong_n','missed_n','no_action_n',
               'correct_candidate_n','retained_correct_candidate_n')


def ratio(a,b):
    return a/b if b else None


def reports(counts):
    n = dict(zip(COUNT_NAMES,map(int,counts)))
    return dict(n,correct_rate=ratio(n['correct_n'],n['move_n']),false_rate=ratio(n['false_n'],n['hold_n']),
                wrong_rate=ratio(n['wrong_n'],n['move_n']),
                no_action_rate=ratio(n['no_action_n'],n['move_n']+n['hold_n']),
                candidate_correct_rate=ratio(n['correct_candidate_n'],n['move_n']),
                conditional_retention=ratio(n['retained_correct_candidate_n'],n['correct_candidate_n']))


def indexed_curve(rows, name, indices):
    selected = [i for i,r in enumerate(rows) if r['confidence'][name] is not None]
    selected.sort(key=lambda i:-rows[i]['confidence'][name])
    scores = np.array([rows[i]['confidence'][name] for i in selected])
    end = np.flatnonzero(np.r_[scores[:-1] != scores[1:],True]) if len(scores) else np.array([],dtype=int)
    groups = indices[selected]
    positives = np.array([rows[i]['candidate_correct'] for i in selected],dtype=int)
    false = np.array([not rows[i]['needs_correction'] for i in selected],dtype=int)
    return groups,positives,false,end,scores[end]


def curve_point(curve, weights, moving, holding):
    groups,positives,false,end,_ = curve
    w = weights[groups]
    correct = np.r_[0,np.cumsum(w*positives)[end]] / moving
    bad = np.r_[0,np.cumsum(w*false)[end]] / holding
    ix = int(np.searchsorted(correct,TARGET))
    if ix == len(correct):
        return correct,bad,None
    fraction = (TARGET-correct[ix-1])/(correct[ix]-correct[ix-1])
    point = bad[ix-1]+fraction*(bad[ix]-bad[ix-1])
    return correct,bad,float(point)


def interval(values):
    return np.quantile(values,[.025,.975]).tolist() if len(values) else None


def paired_analysis(rows, n_boot=BOOTSTRAP_N):
    units = sorted({(r['dataset'],r['group']) for r in rows})
    assert len({group for _,group in units}) == len(units)
    key = {u:i for i,u in enumerate(units)}
    indices = np.array([key[(r['dataset'],r['group'])] for r in rows])
    n = len(key)
    balance = {dataset:np.array([i for i,u in enumerate(units) if u[0]==dataset])
               for dataset in sorted({u[0] for u in units})}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    drawn = np.zeros((n_boot,n),dtype=np.int32)
    for ixs in balance.values():
        drawn[:,ixs] = rng.multinomial(len(ixs),np.ones(len(ixs))/len(ixs),size=n_boot)
    all_once = np.ones(n,dtype=int)
    totals = {}
    intervals = {}
    bootstrap = {}
    event_scenes = {}
    for method in METHODS:
        counts = action_counts(rows,method)
        grouped = np.zeros((n,counts.shape[1]),dtype=int)
        np.add.at(grouped,indices,counts)
        totals[method] = reports(grouped.sum(axis=0))
        boot = drawn @ grouped
        rates = np.stack([boot[:,2]/boot[:,0],boot[:,3]/boot[:,1],boot[:,4]/boot[:,0],
                          boot[:,6]/(boot[:,0]+boot[:,1])],axis=1)
        bootstrap[method] = rates
        intervals[method] = {k:interval(v) for k,v in zip(('correct','false','wrong','no_action'),rates.T)}
        event_scenes[method] = int(np.count_nonzero(grouped[:,3]))
    candidate_by_scene = np.zeros((n,3),dtype=int)
    np.add.at(candidate_by_scene, indices, [(r['needs_correction'],not r['needs_correction'],r['candidate_correct'])
                                           for r in rows])
    moving,holding,ceiling = candidate_by_scene.sum(axis=0)
    curves = {name:indexed_curve(rows,name,indices) for name in GATES}
    outputs = {}
    for name,curve in curves.items():
        correct,bad,point = curve_point(curve,all_once,moving,holding)
        outputs[name] = dict(correct=correct.tolist(),false=bad.tolist(),at_80=point)
    paired = []
    for weights,(m,h,c) in zip(drawn,drawn @ candidate_by_scene):
        if c/m < TARGET:
            continue
        points = [curve_point(curve,weights,m,h)[2] for curve in curves.values()]
        assert all(p is not None for p in points)
        paired.append(points[0]-points[1])
    observed = [outputs[name]['at_80'] for name in GATES]
    point = None if None in observed else observed[0]-observed[1]
    return dict(scene_n=n,n_boot=n_boot,correct_candidate_n=int(ceiling),move_n=int(moving),hold_n=int(holding),
        candidate_ceiling=ceiling/moving, curves=outputs, improvement=point,
        improvement_ci=interval(paired),unreachable_bootstrap=1-len(paired)/n_boot,
        fixed=totals,fixed_ci=intervals,false_event_scenes=event_scenes,
        fixed_improvement_ci=interval(bootstrap['single'][:,1]-bootstrap['dual'][:,1]),
        fixed_correct_dual_minus_single_ci=interval(bootstrap['dual'][:,0]-bootstrap['single'][:,0]))


def strata(rows):
    groups = defaultdict(list)
    for r in rows:
        name = r['condition']
        for key in ('all', r['dataset'], name):
            groups[(r['degradation'],key)].append(r)
        if name in CONDITIONS[:4]:
            groups[(r['degradation'],f"global_{r['global_magnitude']:g}px")].append(r)
    return [dict(degradation=kind,stratum=group,methods={
        method:reports(action_counts(subset,method).sum(0)) for method in METHODS})
        for (kind,group),subset in sorted(groups.items())]


def relative_vectors(rows):
    grouped = defaultdict(dict)
    for r in rows:
        grouped[(r['scene'],r['degradation'])][r['condition']] = r
    vectors = defaultdict(list)
    cf = defaultdict(list)
    confusion = defaultdict(Counter)
    for (_,kind),cases in grouped.items():
        for condition in CONDITIONS[:4]:
            r = cases[condition]
            confusion[(kind,condition)][r['attribution']] += 1
            for arm in ('hr','sr'):
                f,base = r[arm+'_fit'],cases['zero'][arm+'_fit']
                error = float(np.max(np.abs(np.asarray(f['vector'])-base['vector']+r[arm+'_added']))) \
                        if f['valid'] and base['valid'] else None
                vectors[(kind,r['dataset'],arm,condition)].append(error)
        for a,b in (('zero','sr_global'),('hr_global','both_global')):
            one,two = cases[a],cases[b]
            assert one['confidence']['single'] == two['confidence']['single']
            for method in GATES:
                cf[(kind,method,a,b)].append((is_move(one['actions'][method]),is_move(two['actions'][method]),
                                            one['candidate_correct']))
    errors = []
    for (kind,dataset,arm,condition),v in sorted(vectors.items()):
        valid = np.array([x for x in v if x is not None])
        errors.append(dict(degradation=kind,dataset=dataset,arm=arm,condition=condition,n=len(v),valid=len(valid),
            within_025=int((valid<=.25).sum()),within_1=int((valid<=1).sum()),
            median=float(np.median(valid)) if len(valid) else None))
    return dict(vector_change=errors, counterfactual=[
        dict(degradation=k,gate=m,a=a,b=b,n=len(v),a_pass=sum(x for x,_,_ in v),b_pass=sum(y for _,y,_ in v),
             newly_passed=sum(not x and y for x,y,_ in v),newly_blocked=sum(x and not y for x,y,_ in v),
             correct_candidate_newly_blocked=sum(x and not y and c for x,y,c in v)) for (k,m,a,b),v in sorted(cf.items())],
        attribution=[dict(degradation=k,truth=t,n=sum(counts.values()),counts=dict(counts),
                         recall=counts[t]/sum(counts.values()),coverage=1-counts['rejected']/sum(counts.values()))
                     for (k,t),counts in sorted(confusion.items())])


def scoring_stats(rows, split):
    groups = defaultdict(list)
    for row in rows:
        for key in ('all',row['condition']):
            groups[(row['degradation'],key)].append(row)
    result = []
    for (kind,group),subset in sorted(groups.items()):
        def abs_err(r,method):
            return abs(r['scores'][method]-r['scores']['oracle'])
        row = {method:float(np.mean([abs_err(r,method) for r in subset])) for method in METHODS}
        units = sorted({(r['dataset'],r['group']) for r in subset})
        by_group = defaultdict(list)
        for r in subset:
            by_group[(r['dataset'],r['group'])].append(r)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        drawn = []
        for dataset in sorted({u[0] for u in units}):
            units_ds = [u for u in units if u[0]==dataset]
            drawn.append(rng.multinomial(len(units_ds),np.ones(len(units_ds))/len(units_ds),size=BOOTSTRAP_N))
        bs = np.column_stack(drawn)
        count = np.array([len(by_group[u]) for u in units],dtype=float)
        n_boot = bs @ count
        errors = np.array([[sum(abs_err(r,m) for r in by_group[u]) for m in METHODS] for u in units],dtype=float)
        boot = (bs @ errors)/n_boot[:,None]
        clean = [abs(r['scores']['oracle']-r['scores']['clean_hr']) for r in subset if 'clean_hr' in r['scores']]
        result.append(dict(degradation=kind,stratum=group,n=len(subset),mae=row,
                           ci={m:interval(boot[:,i]) for i,m in enumerate(METHODS)},
                           oracle_clean_n=len(clean),oracle_clean_mae=float(np.mean(clean)) if clean else None))
    dump(HERE/split/'shafe_analysis.json',dict(rows=result,bootstrap_n=BOOTSTRAP_N))


def run(split, scoring=False):
    if split in ('formal','external'):
        from freeze import verify
        frozen = verify()
    base = HERE/split
    geometry = load(base/'geometry.json')
    rows = attach(geometry['rows'],load(HERE/'calibration/gates.json'))
    if split=='formal':
        present = {(r['scene'],r['degradation'],r['condition']) for r in rows}
        expected = {(r['scene'],kind,condition) for r in frozen['injection_plan']
                    for kind in ('normal','strong') for condition in CONDITIONS}
        if present != expected or len(rows)!=frozen['formal_cases']:
            raise ValueError('Formal measurement does not match the frozen case set')
    diagnostics = relative_vectors(rows)
    report = dict(split=split,diagnostics=diagnostics)
    if split != 'external':
        report['groups'] = strata(rows)
        report['primary'] = paired_analysis([r for r in rows if r['degradation']=='strong'])
    dump(base/'decisions.json',dict(rows=rows,gates_sha256=sha(HERE/'calibration/gates.json')))
    dump(base/'evaluation.json',report)
    if scoring:
        scored = load(base/'scores.json')
        configuration = scored['scoring']
        if (configuration['geometry_sha256']!=sha(base/'geometry.json') or
                configuration['gates_sha256']!=sha(HERE/'calibration/gates.json')):
            raise ValueError('SHAFE result provenance drift')
        key = lambda r:(r['scene'],r['degradation'],r['condition'])
        index = {key(r):r for r in rows}
        score_rows = scored['rows']
        if len(score_rows)!=len(rows) or {key(r) for r in score_rows}!=set(index):
            raise ValueError('Incomplete SHAFE score set')
        if any(r['actions']!=index[key(r)]['actions'] for r in score_rows):
            raise ValueError('SHAFE scoring actions differ from analysis actions')
        scoring_stats(score_rows,split)
    if 'primary' in report:
        primary=report['primary']
        print({k:primary[k] for k in ('scene_n','correct_candidate_n','move_n','hold_n',
                                      'improvement','improvement_ci','unreachable_bootstrap')},flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('split',choices=('formal','calibration','external'))
    parser.add_argument('--scoring',action='store_true')
    args=parser.parse_args()
    run(args.split,args.scoring)
