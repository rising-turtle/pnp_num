# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:32:58 2019

@author: fuyin

find out the relationship between the accuracy of pnp and the number of points 

"""

import cam_model
import align as al
import numpy as np

def get_pts(cam):
    """
    get feature points 
    """
    w = 10
    s = 10
    px = []
    py = []
    pz = []
    for u in range(w, cam.width - w, s):
        for v in range(w, cam.height - w, s):
            z = np.random.uniform(3.0, 7.0)
            x,y,z = cam.project(u, v, z)
            px.append(x)
            py.append(y)
            pz.append(z)
    pts = np.array([[x, y, z] for (x,y,z) in zip(px, py, pz)]).transpose()
    return pts



def get_pose():
    """
    generate different pose changes
    """
    
    angle = [3, 6, 9] # rotation [degrees] 
    dis = [0.3, 0.6, 0.9] # distance [m]
    
    pos = []
    
    for roll in angle:
        for pitch in angle:
            for yaw in angle:
                for x in dis:
                    for y in dis: 
                        for z in dis:
                          pos.append([x, y, z, roll, pitch, yaw])  
    
    return np.array(pos)

def transform_pts_euler(pts, pose):
    """
    transform points using pose, pts = R*pts + t 
    
    pts [3xn]
    pose [x, y, z, roll, pitch, yaw]
    
    """
    T = al.transform44Euler(pose)
    R = T[0:3,0:3]
    t = T[0:3,3]
    tt_col = pts.shape[1]
    rt_pts = np.zeros((3, tt_col))
    for col in range(tt_col):
        rt_pts[:,col] = R.dot(pts[:,col]) + t
    # pts = np.array([ (R*pt.transpose() + t).transpose() for pt in pts.transpose()])
    
    return rt_pts


def test_pnp_num():
    
    


if __name__=="__main__":
    
    cam = cam_model.structCore()
    pts_j = get_pts(cam)
    Pij = [1., 2., 3., al.d2r(10.), al.d2r(20.), al.d2r(30.)]
    pts_i = transform_pts_euler(pts_j, Pij) 
    Rij, tij, _ = al.align(np.asmatrix(pts_j), np.asmatrix(pts_i))
    rpy = al.rotationMatrixToEulerAngles(Rij)
    print('tij = {}, rpy = {}'.format(tij.transpose(), al.r2d(rpy)))
    
    
    
    
    
    
    
    
    
    


        


