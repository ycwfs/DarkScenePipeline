"""Regression tests for unknown-joint masking and time-based filtering."""
import numpy as np
from prepare import filter_pose, runs


def fixture(n=30):
    points=np.zeros((n,17,3));points[:,:,0]=np.linspace(20,80,17)
    points[:,:,1]=np.linspace(10,90,17);points[:,:,2]=.9
    valid=np.ones((n,17),bool);boxes=np.tile([0,0,100,100],(n,1));times=np.arange(n)/25
    return points,valid,boxes,times


def test_short_tracks_and_joints_are_not_interpolated():
    p,v,b,t=fixture();v[:,9]=False;v[10:12,9]=True
    smooth,keep,_=filter_pose(p,v,b,t,25)
    assert not keep[:,9].any()
    assert keep[:,5].all()


def test_isolated_spike_is_masked():
    p,v,b,t=fixture();p[15,9,0]+=70
    _,keep,stats=filter_pose(p,v,b,t,25)
    assert not keep[15,9]
    assert stats['isolated_spike_joints']>=1


def test_constant_velocity_is_not_delayed_or_removed():
    p,v,b,t=fixture();p[:,:,0]+=np.arange(len(t))[:,None]*2
    smooth,keep,_=filter_pose(p,v,b,t,25)
    assert keep.all()
    assert np.allclose(smooth[1:-1],p[1:-1,:,:2])


def test_gap_separates_short_sequences():
    mask=np.ones(8,bool);times=np.r_[np.arange(4)/25,1+np.arange(4)/25]
    assert runs(mask,times)==[list(range(4)),list(range(4,8))]
    p,v,b,t=fixture(8);_,keep,_=filter_pose(p,v,b,times,25)
    assert not keep.any()


if __name__=='__main__':
    for test in [test_short_tracks_and_joints_are_not_interpolated,test_isolated_spike_is_masked,
                 test_constant_velocity_is_not_delayed_or_removed,test_gap_separates_short_sequences]:
        test()
    print('4 temporal label tests passed')
