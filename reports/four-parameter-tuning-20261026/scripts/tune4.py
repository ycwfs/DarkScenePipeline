"""Joint calibration by exact cumulative threshold counts; no test-driven selection."""
import argparse
import csv
import importlib.util
from common import *

sys.path.insert(0,str(PREV/'scripts'))
spec=importlib.util.spec_from_file_location('old_threshold_tune',PREV/'scripts/tune.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


class Joint:
    def __init__(self):
        self.ev=old.Evaluation();self.rows=self.ev.rows
        d=np.load(OUT/'inputs.npz');self.rgb=d['rgb'];self.sk=d['skeleton'];self.luma=d['luma'];self.support=d['object_support']
        self.gates=d['gates'].tolist();self.o=self.ev.o;self.classes=[i for i in range(10) if i!=self.o and np.any(self.ev.oldval & (self.ev.truth==i))]
        self.base=dict(alpha=.95,gate=30,action=.85,object=.35)
        bp=self.predict(self.base);assert np.array_equal(bp,self.ev.predict(.85,.35)), 'Baseline changed'
        self.minimum={c:int((bp[self.ev.oldval & (self.ev.truth==c)]==c).sum()) for c in self.classes}
        self.baseline_calibration=self.calibration(self.base)

    def values(self,alpha,gate):
        gi=self.gates.index(gate);p=alpha*self.rgb+(1-alpha)*self.sk[gi]
        dark=self.luma<gate;p[dark]=self.rgb[dark]
        top=p.argmax(1);score=p[np.arange(len(p)),top];support=np.full(len(top),np.inf)
        for ci,c in enumerate([self.ev.drink,self.ev.pickup]):
            m=top==c;support[m]=self.support[gi,m,ci]
        return top,score,support

    def predict(self,s):
        top,score,support=self.values(s['alpha'],s['gate'])
        top[(score<s['action'])|(support<s['object'])]=self.o
        return top

    def calibration(self,s):
        p=self.predict(s);e=self.ev;m=e.metrics(p,e.oldval)
        r=dict(s,old_val_other_fp=int((p[e.oldval_other]!=self.o).sum()),
               old_val_positive_correct=int((p[e.oldval_positive]==e.truth[e.oldval_positive]).sum()),
               old_val_macro_f1=m['macro_f1'],online_val_other_fp=int((p[e.onlineval]!=self.o).sum()),
               new_known_drink_fp=int((p[e.new_false]==e.drink).sum()),
               new_true_drink_tp=int((p[e.new_true]==e.drink).sum()),
               new_unreviewed_drink=int((p[e.new_unknown]==e.drink).sum()),
               new_pickup_triggers=int((p[e.new]==e.pickup).sum()))
        for c in self.classes:r['val_tp_'+BEHAVIORS[c]]=int((p[e.oldval & (e.truth==c)]==c).sum())
        r['objective']=.5*(r['old_val_other_fp']/e.oldval_other.sum()+r['new_known_drink_fp']/e.new_false.sum())
        return r

    def key(self,r):
        dist=abs(r['alpha']-.95)/.49+abs(r['gate']-30)/60+abs(r['action']-.85)/.29+abs(r['object']-.35)/.89
        return r['objective'],-r['old_val_macro_f1'],-r['old_val_positive_correct'],-r['new_true_drink_tp'],dist,r['alpha'],r['gate'],r['action'],r['object']

    def surfaces(self,alpha,gate,ts,os):
        e=self.ev;top,score,support=self.values(alpha,gate)
        # NumPy2 production reject_probs compares float32 probabilities to a weak
        # Python scalar, hence action boundaries must also be float32 here.
        ti=np.searchsorted(np.array(ts,dtype=score.dtype),score,side='right');oi=np.searchsorted(np.array(os),support,side='right')
        bins=ti*(len(os)+1)+oi;shape=(len(ts)+1,len(os)+1)
        def count(mask):
            h=np.bincount(bins[mask & (top!=self.o)],minlength=shape[0]*shape[1]).reshape(shape)
            return h[::-1,::-1].cumsum(0).cumsum(1)[::-1,::-1][1:,1:]
        r=dict(old_val_other_fp=count(e.oldval_other),old_val_positive_correct=count(e.oldval_positive & (top==e.truth)),
               online_val_other_fp=count(e.onlineval),new_known_drink_fp=count(e.new_false & (top==e.drink)),
               new_true_drink_tp=count(e.new_true & (top==e.drink)),new_unreviewed_drink=count(e.new_unknown & (top==e.drink)),
               new_pickup_triggers=count(e.new & (top==e.pickup)))
        f1s=[]
        for c in range(10):
            actual=e.oldval & (e.truth==c);n=int(actual.sum())
            if c==self.o:
                tp=n-r['old_val_other_fp'];predn=int(e.oldval.sum())-count(e.oldval)
            else:
                tp=count(actual & (top==c));predn=count(e.oldval & (top==c))
                if c in self.classes:r['val_tp_'+BEHAVIORS[c]]=tp
            if n:f1s.append(2*tp/(n+predn))
        r['old_val_macro_f1']=np.mean(f1s,axis=0)
        r['objective']=.5*(r['old_val_other_fp']/e.oldval_other.sum()+r['new_known_drink_fp']/e.new_false.sum())
        feasible=np.ones_like(r['objective'],bool);b=self.baseline_calibration
        for k in ['old_val_other_fp','online_val_other_fp','new_known_drink_fp','new_unreviewed_drink']:feasible &= r[k]<=b[k]
        for c in self.classes:feasible &= r['val_tp_'+BEHAVIORS[c]]>=self.minimum[c]
        # Do not call an action threshold optimum if it already removes the sole known new true drink.
        feasible &= np.asarray(ts,dtype=score.dtype)[:,None]<=score[e.new_true][0]
        feasible &= int(top[e.new_true][0])==e.drink
        r['feasible']=feasible
        return r

    def full(self,s):
        p=self.predict(s);e=self.ev;out=dict(setting=s,calibration=self.calibration(s))
        for scope in ['onsite','online_supplement']:
            out[scope]={split:e.metrics(p,(e.scope==scope)&((e.split==split) if split!='all' else True)) for split in ['all','train','val','test']}
        from collections import Counter
        out['new']=dict(windows=int(e.new.sum()),known_drink_positive=1,known_drink_negative=526,
            known_drink_tp=int((p[e.new_true]==e.drink).sum()),known_drink_fp=int((p[e.new_false]==e.drink).sum()),
            unknown_drink=int((p[e.new_unknown]==e.drink).sum()),pickup_triggers=int((p[e.new]==e.pickup).sum()),
            actions=dict(Counter(BEHAVIORS[i] for i in p[e.new])))
        return out


def scan(j,alphas,gates,ts,os,tag):
    best=None;true_best=None;fixed_action=None;per_pair=[];tables=[];count=0;feasible_count=0
    def asrow(a,g,r,ti,oi):
        row=dict(alpha=float(a),gate=int(g),action=float(ts[ti]),object=float(os[oi]))
        row.update({k:v[ti,oi].item() for k,v in r.items()});return row
    for g in gates:
        for a in alphas:
            r=j.surfaces(a,g,ts,os);count+=len(ts)*len(os);feasible_count+=int(r['feasible'].sum())
            for target,mask in [('main',r['feasible']),('true',r['feasible']&(r['new_true_drink_tp']==1)),
                                ('fixed',r['feasible']&np.isclose(np.asarray(ts)[:,None],.85,atol=1e-8))]:
                candidates=[asrow(a,g,r,ti,oi) for ti,oi in zip(*np.where(mask))]
                if not candidates:continue
                pick=min(candidates,key=j.key)
                if target=='main':
                    per_pair.append(pick)
                    if best is None or j.key(pick)<j.key(best):best=pick
                elif target=='true':
                    if true_best is None or j.key(pick)<j.key(true_best):true_best=pick
                elif fixed_action is None or j.key(pick)<j.key(fixed_action):fixed_action=pick
            # Full numerical grid retained compactly, including infeasible settings.
            tables.append({k:v for k,v in r.items()})
        print(tag,'gate',g,'done; best',None if best is None else {k:best[k] for k in ['alpha','gate','action','object','objective']},flush=True)
    keys=list(tables[0]);np.savez_compressed(OUT/f'{tag}-grid.npz',alphas=alphas,gates=gates,actions=ts,objects=os,
                                          **{k:np.stack([r[k] for r in tables]) for k in keys})
    if per_pair:old.csv_save(OUT/f'{tag}-best-by-alpha-gate.csv',per_pair)
    return dict(best=best,true_best=true_best,fixed_action=fixed_action,count=count,feasible_count=feasible_count)


def setting(r):return {k:r[k] for k in ['alpha','gate','action','object']}


def main():
    j=Joint();p=json.loads((OUT/'protocol.json').read_text());grid=p['grid']
    coarse=scan(j,grid['alpha'],j.gates,grid['action'],grid['object'],'coarse')
    save(OUT/'coarse-selection-lock.json',dict(protocol_sha256=digest(OUT/'protocol.json'),gate_amendment_sha256=digest(OUT/'gate-coverage-amendment.json'),**coarse))
    c=coarse['best'];assert c
    alphas=[i/1000 for i in range(round(c['alpha']*1000)-10,round(c['alpha']*1000)+11) if 500<=i<=990]
    ts=[i/1000 for i in range(round(c['action']*1000)-10,round(c['action']*1000)+11) if 700<=i<=990]
    os=[i/1000 for i in range(round(c['object']*1000)-10,round(c['object']*1000)+11) if 10<=i<=900]
    # Keep every measured gate in refinement: tied coarse gates must not be silently discarded.
    fine=scan(j,alphas,j.gates,ts,os,'fine')
    chosen=min([coarse['best'],fine['best']],key=j.key)
    true_options=[r for r in [coarse['true_best'],fine['true_best']] if r]
    fixed_options=[r for r in [coarse['fixed_action'],fine['fixed_action']] if r]
    lock=dict(baseline=j.baseline_calibration,per_class_minimum={BEHAVIORS[c]:n for c,n in j.minimum.items()},
              coarse=coarse,fine=fine,recommended=chosen,
              true_drink_option=min(true_options,key=j.key) if true_options else None,
              fixed_action_option=min(fixed_options,key=j.key) if fixed_options else None,
              inputs_sha256=digest(OUT/'inputs.npz'),selection_inputs='Old val and reviewed new binary feedback only.')
    save(OUT/'selection-lock.json',lock)
    settings=dict(baseline=j.base,previous_two_parameter=p['previous_candidate'],recommended=setting(chosen),coarse=setting(coarse['best']))
    if lock['true_drink_option']:settings['true_drink_option']=setting(lock['true_drink_option'])
    if lock['fixed_action_option']:settings['fixed_action_option']=setting(lock['fixed_action_option'])
    # Post-lock diagnosis includes full old train/test and all known losses.
    save(OUT/'summary.json',{name:j.full(s) for name,s in settings.items()})
    save(OUT/'predictions.json',dict(settings=settings,ids=[r['id'] for r in j.rows],labels=BEHAVIORS,
         predictions={name:j.predict(s).tolist() for name,s in settings.items()}))
    save(OUT/'recommended-settings.json',dict(skel_fusion_alpha=chosen['alpha'],lum_gate=chosen['gate'],
         reco_min_conf=chosen['action'],object_min_conf=chosen['object'],skel_no_person='soft',
         status='Offline joint-calibration candidate; not a production package update',model_policy='All six frozen hashes unchanged'))
    print(json.dumps(lock,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
