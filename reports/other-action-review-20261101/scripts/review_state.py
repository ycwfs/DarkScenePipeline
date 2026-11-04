from dataclasses import dataclass,field

@dataclass
class CandidateSequence:
    min_windows:int
    min_span:float
    max_gap:float=.65
    times:list=field(default_factory=list)
    ids:list=field(default_factory=list)
    label:str|None=None
    emitted:bool=False

    def push(self,row:dict,eligible:bool)->dict:
        t=float(row['timestamp']);label=row['raw_label']
        if not eligible:
            self.reset()
            return dict(recovered=False,reason='ineligible')
        if self.times and (t-self.times[-1]>self.max_gap+1e-6 or label!=self.label):
            self.reset()
        if row['id'] in self.ids:
            raise ValueError('duplicate window cannot count twice')
        self.times.append(t);self.ids.append(row['id']);self.label=label
        if len(self.times)<self.min_windows or t-self.times[0]+1e-6<self.min_span:
            return dict(recovered=False,reason='waiting_for_continuity')
        first=not self.emitted;self.emitted=True
        return dict(recovered=True,reason='same_class_temporal_consensus',
            first_confirmation=first,confirmation_delay_s=t-self.times[0] if first else None,
            wait_s=t-self.times[0] if first else 0.,supporting_windows=self.ids[-self.min_windows:])

    def reset(self):
        self.times.clear();self.ids.clear();self.label=None;self.emitted=False
