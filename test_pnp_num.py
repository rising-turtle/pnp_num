# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:32:58 2019

@author: fuyin

find out the relationship between the accuracy of pnp and the number of points 

"""

import cam_model
import align as al
import numpy as np
import pnp_cv 

def add_noise(z):
    """
    std of z is a function of (z)
    since z = f*base_line/disparity;  disparity = f*base_line/z
    """
    bl = 0.1
    f = 500.
    disparity = f*bl/z
    disparity += np.random.randn(1)
    z = f*bl/disparity
    return z

def get_pts(cam, noise = False):
    """
    get feature points [x,y,z]
    """
    w = 10
    s = 10
    px = []
    py = []
    pz = []
    for u in range(w, cam.width - w, s):
        for v in range(w, cam.height - w, s):
            z = np.random.uniform(3.0, 7.0)
            if noise:
                z = add_noise(z)
            if z <= 0.1:
                continue
            x,y,z = cam.project(u, v, z)
            px.append(x)
            py.append(y)
            pz.append(z)
    pts = np.array([[x, y, z] for (x,y,z) in zip(px, py, pz)]).transpose()
    return pts

def get_pts_uvd(cam):
    """
    get feature points in the form of [u,v,d]
    """
    w = 10
    s = 10
    pts = []
    for u in range(w, cam.width - w, s):
        for v in range(w, cam.height - w, s):
            z = np.random.uniform(3.0, 7.0)
            d = cam.fx * cam.l / z 
            # x,y,z = cam.project(u, v, z)
            pts.append([u, v, d])
    # pts = np.array([[x, y, z] for (x,y,z) in zip(px, py, pz)]).transpose()
    pts = np.array(pts).transpose()
    return pts
    
    
def pts_from_uvd(pts, cam, noise = False):
    """
    compute pts [x,y,z] from [u,v,d]
    """
    if noise == False:
        xyz = np.array([cam.project_uvd(uvd[0], uvd[1], uvd[2]) for uvd in pts.transpose()]).transpose()
    else:
        pts = np.array([[uvd[0]+np.random.randn(), uvd[1]+np.random.randn(), uvd[2]+np.random.randn()] for uvd in pts.transpose()]).transpose()
        xyz = np.array([cam.project_uvd(uvd[0], uvd[1], uvd[2]) for uvd in pts.transpose()]).transpose()
    # uv = np.array([(pt[0], pt[1]) for pt in pts.transpose()]).transpose()
    return xyz, pts

def transform_pts_uvd(R, t, cam, pts):
    """
    pts' = R*pts + t, pts in the form of [u, v, d]
    """
    def T(cam, pt, R, t):
        z = cam.fx * cam.l / pt[2]
        x, y, z = cam.project(pt[0], pt[1], z)
        xyz1 = np.array([[x,y,z]]).transpose()
        xyz = np.matmul(R, xyz1) 
        xyz += np.array([t]).transpose()
        u, v = cam.inv_proj(xyz[0], xyz[1], xyz[2])
        d = cam.fx * cam.l / xyz[2]
        return u[0], v[0], d[0]
           
    pts_n = np.array([T(cam, pt, R,t) for pt in pts.transpose()])
    pts = pts.T
    pts_t = np.array([np.hstack((pn, pt)) for (pn, pt) in zip(pts_n, pts) if pn[0] > 0 and pn[0]< cam.width \
                     and pn[1] > 0 and pn[1] < cam.height and pn[2] > 0])
    return pts_t[:,0:3].T, pts_t[:,3:].T
    

def get_pose():
    """
    generate different pose changes
    """
    
    angle = [3, 6, 9] # rotation [degrees] 
    dis = [0.2, 0.4] # distance [m]
    
    pos = []
    
    for roll in angle:
        for pitch in angle:
            for yaw in angle:
                for x in dis:
                    # for y in dis: 
                        for z in dis:
                          pos.append([x, 0., z, al.d2r(roll), al.d2r(pitch), al.d2r(yaw)])  
    
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


def generate_data_for_test():
    cam = cam_model.structCore()
    pos_set = get_pose()

    for pos in pos_set:
         T = al.transform44Euler(pos)
         R = T[0:3,0:3]
         t = T[0:3,3]
         print('gt: tij = {}, rpy = {}'.format(t.transpose(), al.r2d(al.rotationMatrixToEulerAngles(R).transpose())))
         pts_j = get_pts_uvd(cam)
         pts_i, pts_j = transform_pts_uvd(R, t, cam, pts_j)
         pts_j, pts_j_uv = pts_from_uvd(pts_j, cam, noise = True)
         pts_i, pts_i_uv = pts_from_uvd(pts_i, cam, noise = True)
         # Rij, tij, _ = al.align(np.asmatrix(pts_j), np.asmatrix(pts_i))
         
         # Rij, tij = pnp_cv.pnp_2d2d(pts_j_uv[:2,:].transpose(), pts_i_uv[:2,:].transpose(), cam.K)
         Rij, tij = pnp_cv.pnp_3d2d(pts_j.transpose(), pts_i_uv[:2,:].transpose(), cam.K)
         
         rpy = al.rotationMatrixToEulerAngles(Rij)
         print('est: tij = {}, rpy = {}'.format(tij.transpose(), al.r2d(rpy))) 
         

if __name__=="__main__":
    
    generate_data_for_test()
#    cam = cam_model.structCore()
#    
#    pts_j = get_pts_uvd(cam)
#    Pij = [0.1, 0.2, 0.3, al.d2r(0.), al.d2r(2.), al.d2r(0.)]
#    # Pij = [0., 0., 0., al.d2r(0.), al.d2r(0.), al.d2r(0.)]
#    T = al.transform44Euler(Pij)
#    R = T[0:3,0:3]
#    t = T[0:3,3]
#    pts_i, pts_j = transform_pts_uvd(R, t, cam, pts_j)
#    pts_j, pts_j_uv = pts_from_uvd(pts_j, cam, noise = False)
#    pts_i, pts_i_uv = pts_from_uvd(pts_i, cam, noise = False)
#    # Rij, tij, _ = al.align(np.asmatrix(pts_j), np.asmatrix(pts_i))
#    
#    Rij, tij = pnp_cv.pnp_2d2d(pts_j_uv[:2,:].transpose(), pts_i_uv[:2,:].transpose(), cam.K)
#    
#    rpy = al.rotationMatrixToEulerAngles(Rij)
#    print('tij = {}, rpy = {}'.format(tij.transpose(), al.r2d(rpy)))

    
    
    
    
    
    
    
    
    


        


