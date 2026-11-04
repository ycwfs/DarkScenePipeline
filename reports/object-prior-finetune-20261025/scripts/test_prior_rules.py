"""Safety-relevant behavior of the isolated veto, including occlusion and other people."""
import unittest
import numpy as np
from prior_rules import decide


def samples(cls=2, score=.9):
    result=[]
    for t in [0.,.25,.5,.75,1.]:
        pose=np.zeros((17,3));pose[:5]=[50,12,1];pose[9]=[55,20,1];pose[10]=[25,55,1]
        pose[13:15]=[40,75,1]
        result.append(dict(timestamp=t,person_box=[0,0,100,100],pose=pose.tolist(),
            binding_valid=True,detections=[[50,15,60,25,score,cls]]))
    return result


class RulesTests(unittest.TestCase):
    def test_phone_at_head_rejects_drink_only(self):
        s=samples();self.assertEqual(decide('drink',s)['conservative'],'other')
        self.assertEqual(decide('pick_up',s)['conservative'],'pick_up')

    def test_missing_keeps_conservative(self):
        s=samples()
        for row in s:row['detections']=[]
        self.assertEqual(decide('drink',s)['conservative'],'drink')
        self.assertEqual(decide('drink',s)['strict'],'other')

    def test_single_transient_and_wrong_person_do_not_veto(self):
        s=samples()
        for row in s[1:]:row['detections']=[]
        self.assertEqual(decide('drink',s)['conservative'],'drink')
        s=samples()
        for row in s:row['binding_valid']=False
        self.assertEqual(decide('drink',s)['conservative'],'drink')

    def test_occlusion_breaks_confirmation(self):
        s=samples();s[2]['detections']=[]
        self.assertEqual(decide('drink',s)['conservative'],'drink')

    def test_container_in_hand_at_mouth_supports(self):
        self.assertEqual(decide('drink',samples(cls=0))['strict'],'drink')
        s=samples(cls=0)
        for row in s:row['detections'][0][:4]=[10,70,20,80]
        self.assertEqual(decide('drink',s)['strict'],'other')

    def test_no_action_promotion(self):
        for label in ['other','wave','chase']:
            self.assertEqual(decide(label,samples())['strict'],label)

    def test_stationary_ground_object_not_pickup(self):
        s=samples(cls=4)
        for row in s:
            row['detections'][0][:4]=[45,80,55,90];row['pose'][9]=[50,85,1]
        self.assertEqual(decide('pick_up',s)['strict'],'other')
        for i,row in enumerate(s):
            y=85-i*7;row['detections'][0][:4]=[45,y-5,55,y+5];row['pose'][9]=[50,y,1]
        self.assertEqual(decide('pick_up',s)['strict'],'pick_up')

    def test_exhibit_requires_actor_overlap(self):
        s=samples(cls=6)
        for row in s:row['detections'][0][:4]=[0,0,100,100]
        self.assertEqual(decide('drink',s)['conservative'],'other')
        for row in s:row['detections'][0][:4]=[70,0,170,100]
        self.assertEqual(decide('drink',s)['conservative'],'drink')


if __name__=='__main__':unittest.main()
