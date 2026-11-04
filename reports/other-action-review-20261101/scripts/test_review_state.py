import unittest
from review_state import CandidateSequence

def row(i,t,label='drink'):
    return dict(id=str(i),timestamp=t,raw_label=label)

class SequenceTests(unittest.TestCase):
    def test_two_window_confirmation_and_delay(self):
        s=CandidateSequence(2,.5)
        self.assertFalse(s.push(row(1,0),True)['recovered'])
        r=s.push(row(2,.52),True)
        self.assertTrue(r['recovered']);self.assertAlmostEqual(r['confirmation_delay_s'],.52)
        self.assertEqual(s.push(row(3,1.04),True)['wait_s'],0.)
    def test_three_windows_require_one_second(self):
        s=CandidateSequence(3,1.)
        s.push(row(1,0),True);s.push(row(2,.52),True)
        r=s.push(row(3,1.04),True)
        self.assertTrue(r['recovered']);self.assertAlmostEqual(r['confirmation_delay_s'],1.04)
    def test_prior_failure_resets(self):
        s=CandidateSequence(2,.5);s.push(row(1,0),True);s.push(row(2,.52),False)
        self.assertFalse(s.push(row(3,1.04),True)['recovered'])
    def test_class_change_resets(self):
        s=CandidateSequence(2,.5);s.push(row(1,0,'drink'),True);s.push(row(2,.52,'pick_up'),True)
        self.assertFalse(s.emitted)
    def test_gap_resets(self):
        s=CandidateSequence(2,.5);s.push(row(1,0),True);s.push(row(2,.66),True)
        self.assertFalse(s.emitted)
    def test_duplicate_does_not_count_twice(self):
        s=CandidateSequence(2,.5);s.push(row(1,0),True)
        with self.assertRaises(ValueError):s.push(row(1,.52),True)

if __name__=='__main__':unittest.main()
