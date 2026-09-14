"""Attention metrics with fixed supports, equal windows, and explicit selectors."""
import numpy as np


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    total = x.sum(axis=-1, keepdims=True)
    return np.divide(x, total, out=np.zeros_like(x), where=total > 0)


def js(p, q):
    p, q = normalize(p), normalize(q)
    if p.sum() == 0 or q.sum() == 0:
        return None
    m = (p+q)/2
    def term(a):
        mask = a > 0
        return float(np.sum(a[mask]*np.log2(a[mask]/m[mask])))
    return max(0., (term(p)+term(q))/2)


def kl(p, q, eps=1e-9):
    p, q = normalize(p), normalize(q)
    if p.sum() == 0 or q.sum() == 0:
        return None
    p, q = normalize(p+eps), normalize(q+eps)
    return max(0., float(np.sum(p*np.log2(p/q))))


def windows(mass, size, stride=None):
    stride = size if stride is None else stride
    return [mass[i:i+size].mean(0) for i in range(0, len(mass)-size+1, stride)]


def mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def bootstrap(xs, seed=20260911, n=5000):
    xs = np.asarray([x for x in xs if x is not None and np.isfinite(x)])
    if not len(xs):
        return {'n': 0, 'mean': None, 'ci95': [None,None]}
    if len(xs)==1:
        v=float(xs[0])
        return {'n':1,'mean':v,'median':v,'p10':v,'p90':v,'ci95':[None,None]}
    rng = np.random.default_rng(seed)
    means = xs[rng.integers(len(xs), size=(n,len(xs)))].mean(1)
    return {'n':len(xs), 'mean':float(xs.mean()), 'median':float(np.median(xs)),
            'p10':float(np.quantile(xs,.1)), 'p90':float(np.quantile(xs,.9)),
            'ci95':[float(v) for v in np.quantile(means,[.025,.975])]}


def frozen_curves(mass, scores, ks=(8,16,32,64), bins=10):
    """Scores must come from information available before the evaluated rows."""
    p = normalize(mass)
    valid = mass.sum(1)>0
    if not valid.any() or scores.sum()<=0:
        return {}
    order = np.argsort(-scores, kind='stable')
    curves = {}
    targets=[(str(k),k) for k in ks]
    targets += [(f'r{int(frac*100):02d}',max(1,int(np.ceil(frac*p.shape[1])))) for frac in [.05,.1,.2]]
    for name,k in targets:
        retained = p[:,order[:k]].sum(1)
        oracle = np.sort(p,axis=1)[:,-min(k,p.shape[1]):].sum(1)
        boundaries = np.linspace(0,len(p),bins+1,dtype=int)
        def bin_means(v):
            return [mean(v[a:b][valid[a:b]]) for a,b in zip(boundaries[:-1],boundaries[1:])]
        curves[name] = dict(frozen=bin_means(retained),oracle=bin_means(oracle),
            gap=bin_means(oracle-retained),mean=mean(retained[valid]),
            p10=float(np.quantile(retained[valid],.1)),
            oracle_mean=mean(oracle[valid]),candidate_blocks=p.shape[1],selected_blocks=min(k,p.shape[1]))
    return curves


def analyze_steps(steps, window=64, stride=None, sink=1, recent=8,
                  identity_verified=False, compute_coverage=True):
    """steps contain matrices + prompt_tokens; prefix limit may be exact or positional."""
    within_js, within_kl, cross_js, cross_kl, details = [],[],[],[],[]
    matched=[]
    prev = None
    for i, step in enumerate(steps):
        mass, pos = step['mass'], step['positions']
        bt = step.get('block_tokens',128)
        prompt = step['prompt_tokens']
        decode = mass[pos >= prompt]
        # Fully old blocks; exclude a partial prompt block and recent band.
        history_end=step.get('history_end_tokens',prompt)
        end = min(mass.shape[1],max(sink,min(prompt//bt-recent,history_end//bt)))
        hist = decode[:,sink:end]
        ws = windows(hist,window,stride) if end>sink else []
        wjs = [js(a,b) for a,b in zip(ws,ws[1:])]
        wkl = [kl(a,b) for a,b in zip(ws,ws[1:])]
        within_js.append(mean(wjs));within_kl.append(mean(wkl))
        item = dict(step=step.get('step',i),prompt_tokens=prompt,
            decode_samples=len(decode),full_windows=len(ws),within_pairs=len(wjs),
            within_js=mean(wjs),within_kl=mean(wkl),
            historical_mass_mean=mean(hist.sum(1)) if end>sink else None,
            candidate_blocks=max(0,end-sink))
        if (prev is not None and step.get('step',i)==prev['step']+1 and
                (not identity_verified or step.get('prefix_identity_verified',False))):
            # This limit is exact only when caller supplies a verified prefix.
            common = step.get('common_prefix_tokens',min(prompt,prev['prompt_tokens']))
            upper = min(end,prev['end'],common//bt)
            if upper>sink and len(decode)>=window and len(prev['decode'])>=window:
                a=prev['decode'][-window:,sink:upper].mean(0)
                b=decode[:window,sink:upper].mean(0)
                cjs,ckl=js(a,b),kl(a,b)
                cross_js.append(cjs);cross_kl.append(ckl)
                item.update(cross_js=cjs,cross_kl=ckl,common_blocks=upper-sink,
                            prefix_identity_verified=identity_verified)
                # Same two steps, same support, same window width and overlap.
                if len(decode)>=2*window and len(prev['decode'])>=2*window:
                    before=prev['decode'][-2*window:-window,sink:upper].mean(0)
                    after=decode[window:2*window,sink:upper].mean(0)
                    matched.append(dict(from_step=prev['step'],to_step=step.get('step',i),
                        within_js=mean([js(before,a),js(b,after)]),cross_js=cjs,
                        within_kl=mean([kl(before,a),kl(b,after)]),cross_kl=ckl,
                        common_blocks=upper-sink))
        if end>sink and compute_coverage:
            prefill = mass[pos<prompt,sink:end]
            if len(prefill):
                scores = normalize(prefill).mean(0)
                item['prefill_diagnostic']=frozen_curves(hist,scores)
                item['prefill_query_samples']=len(prefill)
            # Held-out decode proxy for legacy traces, deliberately separate.
            if len(hist)>32:
                item['early_decode_proxy']=frozen_curves(hist[32:],normalize(hist[:32]).mean(0))
        details.append(item)
        # A boundary's support is determined by its own two requests. Do not
        # carry the previous boundary's smaller prefix into a third request.
        prev=dict(step=step.get('step',i),decode=decode,
                  end=min(mass.shape[1],max(sink,prompt//bt-recent)),prompt_tokens=prompt)
    return dict(window=window,stride=window if stride is None else stride,
        within_js=mean(within_js),within_kl=mean(within_kl),cross_js=mean(cross_js),
        cross_kl=mean(cross_kl),within_steps=sum(x is not None for x in within_js),
        cross_boundaries=sum(x is not None for x in cross_js),details=details,
        prefix_identity_verified=identity_verified,matched_boundaries=matched,
        matched_within_js=mean([r['within_js'] for r in matched]),
        matched_cross_js=mean([r['cross_js'] for r in matched]))
