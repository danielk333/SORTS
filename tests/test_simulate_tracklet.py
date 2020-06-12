import sys
import os
sys.path.insert(0,os.path.abspath('.'))
import time
import unittest
import numpy as n
import numpy.testing as nt
import orbital_estimation
import simulate_tracklet
import glob
import h5py
import re

def test_tracklets():
    # Unit test
    #
    # Create tracklets and perform orbit determination
    #
    import population_library as plib
    import radar_library as rlib
    import simulate_tracking
    import simulate_tracklet as st
    import os
    
    os.system("rm -Rf /tmp/test_tracklets")
    os.system("mkdir /tmp/test_tracklets")    
    m = plib.master_catalog(sort=False)

    # Envisat
    o = m.get_object(145128)
    print(o)

    e3d = rlib.eiscat_3d(beam='gauss')

    # time in seconds after mjd0
    t_all = n.linspace(0, 24*3600, num=1000)
    
    passes, _, _, _, _ = simulate_tracking.find_pass_interval(t_all, o, e3d)
    print(passes)

    for p in passes[0]:
        # 100 observations of each pass
        mean_t=0.5*(p[1]+p[0])
        print("duration %1.2f"%(p[1]-p[0]))
        if p[1]-p[0] > 10.0:
            t_obs=n.linspace(mean_t-10,mean_t+10,num=10)
            print(t_obs)
            meas, fnames, ecef_stdevs = st.create_tracklet(o, e3d, t_obs, hdf5_out=True, ccsds_out=True, dname="/tmp/test_tracklets")
    
    fl=glob.glob("/tmp/test_tracklets/*")
    for f in fl:
        print(f)
        fl2=glob.glob("%s/*.h5"%(f))
        print(fl2)
        fl2.sort()
        start_times=[]
        for f2 in fl2:
            start_times.append(re.search("(.*/track-.*)-._..h5",f2).group(1))
        start_times=n.unique(start_times)
        print("n_tracks %d"%(len(start_times)))
        
        for t_pref in start_times[0:1]:
            fl2 = glob.glob("%s*.h5"%(t_pref))
            n_static=len(fl2)
            if n_static == 3:
                print("Fitting track %s"%(t_pref))

                f0="%s-0_0.h5"%(t_pref)
                f1="%s-0_1.h5"%(t_pref)
                f2="%s-0_2.h5"%(t_pref)                

                print(f0)
                print(f1)
                print(f2)                
                
                h0=h5py.File(f0,"r")
                h1=h5py.File(f1,"r")
                h2=h5py.File(f2,"r") 
                
                r_meas0=h0["m_range"].value
                rr_meas0=h0["m_range_rate"].value    
                r_meas1=h1["m_range"].value
                rr_meas1=h1["m_range_rate"].value    
                r_meas2=h2["m_range"].value
                rr_meas2=h2["m_range_rate"].value

                n_t=len(r_meas0)
                if len(r_meas1) != n_t or len(r_meas2) != n_t:
                    print("non-overlapping measurements, tbd, align measurement")
                    continue
            
                p_rx=n.zeros([3,3])
                p_rx[:,0]=h0["rx_loc"].value/1e3
                p_rx[:,1]=h1["rx_loc"].value/1e3
                p_rx[:,2]=h2["rx_loc"].value/1e3    
    
                for ti in range(n_t):
                    if h0["m_time"][ti] != h1["m_time"][ti] or h2["m_time"][ti] != h0["m_time"][ti]:
                        print("non-aligned measurement")
                        continue
                    m_r=n.array([r_meas0[ti],r_meas1[ti],r_meas2[ti]])
                    m_rr=n.array([rr_meas0[ti],rr_meas1[ti],rr_meas2[ti]])
                    ecef_state=orbital_estimation.estimate_state(m_r,m_rr,p_rx)
                    true_state=h0["true_state"].value[ti,:]
                    r_err=1e3*n.linalg.norm(ecef_state[0:3]-true_state[0:3])
                    v_err=1e3*n.linalg.norm(ecef_state[3:6]-true_state[3:6])
                    print("pos error %1.3f (m) vel error %1.3f (m/s)"%(1e3*n.linalg.norm(ecef_state[0:3]-true_state[0:3]),1e3*n.linalg.norm(ecef_state[3:6]-true_state[3:6])))                    
                    assert r_err < 100.0
                    assert v_err < 50.0                    


                h0.close()
                h1.close()
                h2.close()

    os.system("rm -Rf /tmp/test_tracklets")                


