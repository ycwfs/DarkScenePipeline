import unittest
from policy import Continuation,RunningReview

def row(t,score=.8,label='chase',support=True,baseline='other'):
    return dict(id=str(t),timestamp=t,score=score,raw_label=label,support=support,baseline=baseline)

class PolicyTests(unittest.TestCase):
    def test_continues_but_does_not_refresh_anchor(self):
        c=Continuation();c.push(row(0,.95,baseline='chase'))
        self.assertTrue(c.push(row(.52))['recovered']);self.assertTrue(c.push(row(1.04))['recovered'])
        self.assertFalse(c.push(row(1.56))['recovered'])
    def test_other_class_gap_and_prior_reset(self):
        for bad in [row(.52,label='other'),row(.52,label='wave'),row(.52,support=False),row(.52,score=.65),row(.7)]:
            c=Continuation();c.push(row(0,.95,baseline='chase'));c.push(bad)
            self.assertFalse(c.push(row(1.04))['recovered'])
    def test_rejected_high_does_not_anchor(self):
        c=Continuation();c.push(row(0,.95,support=False));self.assertFalse(c.push(row(.52))['recovered'])
    def test_only_original_anchor(self):
        c=Continuation();self.assertFalse(c.push(row(0))['recovered']);self.assertFalse(c.push(row(.52))['recovered'])
    def test_confirm_two_and_no_retroactive_recovery(self):
        r=RunningReview();first=r.push(row(0));second=r.push(row(.52));third=r.push(row(1.04))
        self.assertFalse(first['recovered']);self.assertTrue(second['recovered']);self.assertAlmostEqual(second['wait_s'],.52)
        self.assertEqual(third['wait_s'],0);self.assertFalse(third['first_confirmation'])
    def test_span_required(self):
        r=RunningReview();r.push(row(0));self.assertFalse(r.push(row(.2))['recovered']);self.assertTrue(r.push(row(.52))['recovered'])
    def test_run_reset(self):
        for bad in [row(.52,support=False),row(.52,label='wave'),row(.52,score=.65),row(.52,.95,baseline='chase')]:
            r=RunningReview();r.push(row(0));r.push(bad);self.assertFalse(r.push(row(1.04))['recovered'])
    def test_no_gap_bridge(self):
        r=RunningReview();r.push(row(0));self.assertFalse(r.push(row(.7))['recovered'])
    def test_independent_person_and_generation(self):
        states={};get=lambda key:states.setdefault(key,Continuation())
        get(('a',1,0)).push(row(0,.95,baseline='chase'))
        for key in [('a',2,0),('a',1,1),('b',1,0)]:self.assertFalse(get(key).push(row(.52))['recovered'])
    def test_no_duplicate_evidence(self):
        r=RunningReview();r.push(row(0))
        with self.assertRaises(ValueError):r.push(row(0))
    def test_three_window_diagnostic(self):
        r=RunningReview(3,1.);r.push(row(0));self.assertFalse(r.push(row(.52))['recovered'])
        self.assertAlmostEqual(r.push(row(1.04))['wait_s'],1.04)

if __name__=='__main__':unittest.main()
