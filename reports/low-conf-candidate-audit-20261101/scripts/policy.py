"""Experimental causal policies. No production import or mutation."""
from dataclasses import dataclass,field

@dataclass
class Continuation:
    anchor_time: float | None=None
    anchor_label: str | None=None
    previous_time: float | None=None
    def push(self,r:dict)->dict:
        t=r['timestamp'];out=dict(recovered=False,wait_s=0.,reason='not_continued')
        if self.previous_time is not None and t-self.previous_time>.650001:self.anchor_time=None
        self.previous_time=t
        label=r['raw_label']
        if r['baseline']!='other':
            self.anchor_time=t;self.anchor_label=r['baseline'];out['reason']='original_confirmation';return out
        if label=='other' or r['score']<=.65 or r['score']>=.925 or not r['support']:
            self.anchor_time=None;out['reason']='reset_by_class_confidence_or_prior';return out
        if self.anchor_time is None:return out
        if label!=self.anchor_label or t-self.anchor_time>1.200001:
            self.anchor_time=None;out['reason']='anchor_expired_or_different_class';return out
        return dict(recovered=True,wait_s=0.,reason='same_confirmed_action',anchor_time=self.anchor_time)

@dataclass
class RunningReview:
    min_windows:int=2
    min_span:float=.5
    times:list=field(default_factory=list)
    ids:list=field(default_factory=list)
    confirmed:bool=False
    def push(self,r:dict)->dict:
        t=r['timestamp'];out=dict(recovered=False,wait_s=None,reason='not_running_candidate')
        eligible=r['raw_label']=='chase' and .65<r['score']<.925 and r['support']
        if not eligible:self.times=[];self.ids=[];self.confirmed=False;return out
        if self.times and t-self.times[-1]>.650001:self.times=[];self.ids=[];self.confirmed=False
        if self.ids and self.ids[-1]==r['id']:raise ValueError('duplicate window cannot count twice')
        self.times.append(t);self.ids.append(r['id'])
        if len(self.times)<self.min_windows or t-self.times[0]+1e-6<self.min_span:
            out['reason']='waiting_for_continuity';return out
        first=not self.confirmed;self.confirmed=True
        return dict(recovered=True,wait_s=t-self.times[0]if first else 0.,reason='continuous_reliable_running',
            first_confirmation=first,episode_id=self.ids[0],episode_start=self.times[0],confirmation_delay_s=t-self.times[0]if first else None,
            supporting_windows=self.ids[-self.min_windows:])
