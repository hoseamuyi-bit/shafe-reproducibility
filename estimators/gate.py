"""Fit the two logistic gates on executable development candidates; calibrate once."""
from collections import Counter
import sys

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dev_pipeline import HERE, OLD, dump, sha
from experiment import CONDITIONS, load
from registration import HR_FEATURES

TARGET = .80
METHODS = ('raw','direct','simple_lr','single','dual','legacy','oracle')
ZERO = [0.,0.]


def row_features(row, dual):
    hr, sr = row['hr_fit'], row['sr_fit']
    x = [float(hr[name]) for name in HR_FEATURES]
    if not dual:
        return x
    if not sr['valid']:
        return x + [np.nan] * (len(HR_FEATURES)+4) + [1.]
    delta = np.asarray(sr['vector']) - hr['vector']
    return x + [float(sr[name]) for name in HR_FEATURES] + [
        *delta, float(np.linalg.norm(delta)), float(np.max(np.abs(delta))), 0.]


def argsort_thresholds(scores, correct, false):
    """Group equal scores into identical decisions, keeping the reject-all point."""
    order = np.argsort(-scores,kind='stable')
    ordered = scores[order]
    endings = np.flatnonzero(np.r_[ordered[:-1] != ordered[1:], True])
    return (order,endings, np.r_[0,np.cumsum(correct[order])[endings]],
            np.r_[0,np.cumsum(false[order])[endings]])


def calibrate_threshold(scores, rows):
    moving_count = sum(r['needs_correction'] for r in rows)
    selected = [i for i,r in enumerate(rows) if r['hr_fit']['executable']]
    if not selected:
        return dict(threshold=None,policy='all_executable',reachable=False,correct_n=0,
                    moving_n=moving_count,false_n=0,correct_rate=0.)
    cases = [rows[i] for i in selected]
    values = scores[selected]
    correct = np.array([r['candidate_correct'] for r in cases],dtype=int)
    false = np.array([not r['needs_correction'] for r in cases],dtype=int)
    order, ends, cum_correct, cum_false = argsort_thresholds(values,correct,false)
    reachable = np.flatnonzero(cum_correct >= TARGET*moving_count)
    point=int(reachable[0]) if len(reachable) else len(cum_correct)-1
    return dict(threshold=float(values[order[ends[point-1]]]) if len(reachable) else None,
                policy='score_at_least_threshold' if len(reachable) else 'all_executable',
                reachable=bool(len(reachable)), correct_n=int(cum_correct[point]), moving_n=moving_count,
                false_n=int(cum_false[point]), correct_rate=float(cum_correct[point]/moving_count))


def fit_legacy(calibration):
    sys.path.insert(0,str(OLD/'work/shafe_pilot'))
    from lr_anchor import Registration, calibrate
    limits = {}
    for arm in ('hr','sr'):
        controls = []
        for row in calibration:
            fit = row[arm+'_fit']
            if not fit['valid'] or row['condition'] not in CONDITIONS[:4]:
                continue
            rmse = np.sqrt(fit['residual'])
            runner = np.sqrt(fit['residual']+fit['independent_gap'])
            result = Registration(tuple(fit['vector']),tuple(fit['independent_vector']),rmse,runner,
                                  runner-rmse,fit['boundary'],(12,12,116,116))
            # Historical rule trained on injections defined relative to the corresponding base.
            expected = -np.asarray(row[arm+'_added'])
            label = result.shift if np.max(np.abs(np.asarray(result.shift)-expected)) <= .25 else None
            controls.append((result,label))
        gate = calibrate(controls)
        limits[arm] = {k:float(v) if v is not None else None
                       for k,v in vars(gate).items()}
    return limits


def confidence(row, model, name):
    if not row['hr_fit']['executable']:
        return None
    raw = np.array(row_features(row,name=='dual'))
    x = (np.where(np.isfinite(raw),raw,model['median'])-model['mean'])/model['scale']
    return float(expit(x @ model['coef'] + model['intercept']))


def legacy_status(fit, limits):
    if not fit['valid']:
        return 'invalid_projection'
    if limits['max_rmse'] is None:
        return 'calibration_unavailable'
    if fit['boundary']:
        return 'search_boundary'
    rmse = float(np.sqrt(fit['residual']))
    margin = float(np.sqrt(fit['residual']+fit['independent_gap']))-rmse
    if rmse > limits['max_rmse']:
        return 'residual_above_calibration'
    return 'accepted' if margin >= limits['min_margin'] else 'weak_margin'


def actions(row, frozen):
    fit = row['hr_fit']
    fields = dict(raw=ZERO, direct=row['direct_vector'], simple_lr=fit['vector'] if fit['executable'] else ZERO,
                  oracle=[-x for x in row['hr_added']])
    scores = {}
    reasons = {}
    for name,model in frozen['models'].items():
        value = confidence(row,model,name)
        scores[name] = value
        threshold = model['calibration']['threshold']
        accept = value is not None and (threshold is None or value >= threshold)
        fields[name] = fit['vector'] if accept else ZERO
        reasons[name] = ('executed' if accept else 'below_frozen_threshold' if fit['executable']
                         else 'candidate_within_hold_band' if fit['valid'] else fit['reason'])
    limits = frozen['legacy']
    statuses = {side:legacy_status(row[side+'_fit'],limits[side]) for side in ('hr','sr')}
    detected = (fit['executable'],row['sr_fit']['executable'])
    attribution = {(False,False):'zero',(True,False):'hr_global',
                   (False,True):'sr_global',(True,True):'both_global'}[detected]
    if set(statuses.values()) != {'accepted'}:
        attribution = 'rejected'
    fields['legacy'] = fit['vector'] if attribution == 'hr_global' else ZERO
    return fields,scores,reasons,attribution,statuses


def fit_and_calibrate():
    dpath = HERE/'dev/geometry.json'
    cpath = HERE/'calibration/geometry.json'
    dev = load(dpath)['rows']
    calibration = [r for r in load(cpath)['rows'] if r['degradation']=='strong']
    eligible = [r for r in dev if r['hr_fit']['executable']]
    assert not ({r['group'] for r in dev} & {r['group'] for r in calibration})
    y = [r['candidate_correct'] for r in eligible]
    models = {}
    for name in ('single','dual'):
        dual = name=='dual'
        x = np.array([row_features(r,dual) for r in eligible])
        med = np.nanmedian(x,axis=0)
        med = np.where(np.isfinite(med),med,0.)
        imputed = np.where(np.isfinite(x),x,med)
        standard = StandardScaler().fit(imputed)
        model = LogisticRegression(C=1,solver='lbfgs',tol=1e-10,max_iter=2000)
        model.fit(standard.transform(imputed),y)
        fit = dict(median=med.tolist(),mean=standard.mean_.tolist(),scale=standard.scale_.tolist(),
                   coef=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),iterations=int(model.n_iter_[0]))
        scores = np.array([np.nan if (s:=confidence(r,fit,name)) is None else s for r in calibration])
        fit['calibration'] = calibrate_threshold(scores,calibration)
        models[name] = fit
    legacy = fit_legacy(calibration)
    result = dict(status='CALIBRATED; NO FORMAL OUTCOMES ACCESSED', gate_code_sha256=sha(__file__),
                  development_geometry_sha256=sha(dpath), calibration_geometry_sha256=sha(cpath),
                  train_n=len(eligible),train_positive=sum(y), train_negative=len(y)-sum(y),
                  hr_features=list(HR_FEATURES),
                  additional_dual_features=[*('sr_'+x for x in HR_FEATURES),
                    'sr_minus_hr_dx','sr_minus_hr_dy','sr_minus_hr_l2','sr_minus_hr_linf','sr_invalid'],
                  legacy=legacy,models=models)
    dump(HERE/'calibration/gates.json',result)
    print({name:fit['calibration'] for name,fit in models.items()},flush=True)


if __name__=='__main__':
    fit_and_calibrate()
