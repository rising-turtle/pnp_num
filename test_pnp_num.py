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
import matplotlib.pyplot as plt
import lsq_t_with_R as lsq_tR

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
    s = 20
    px = []
    py = []
    pz = []
    # tmp = [3, 4, 5, 6, 7, 8, 9, 10]
    # i = 0
    for u in range(w, cam.width - w, s):
        for v in range(w, cam.height - w, s):
            z = np.random.uniform(2.1, 5.0)
            # z = tmp[i]*100
            # i = (i+1)%8
            if noise:
                z = add_noise(z)
            if z <= 0.1:
                continue
            x,y,z = cam.project(u, v, z)
            x, y, z = cam.inv_proj_uvd(x, y, z)
            px.append(x)
            py.append(y)
            pz.append(z)
    pts = np.array([[x, y, z] for (x,y,z) in zip(px, py, pz)]).transpose()
    return pts

#def get_pts_uvd(cam):
#    """
#    get feature points in the form of [u,v,d]
#    """
#    w = 10
#    s = 10
#    pts = []
#    for u in range(w, cam.width - w, s):
#        for v in range(w, cam.height - w, s):
#            z = np.random.uniform(3.0, 7.0)
#            d = cam.fx * cam.l / z 
#            # x,y,z = cam.project(u, v, z)
#            pts.append([u, v, d])
#    # pts = np.array([[x, y, z] for (x,y,z) in zip(px, py, pz)]).transpose()
#    pts = np.array(pts).transpose()
#    return pts

def get_pts_uvd(cam, N = 500):
    """
    evenly distribute points in the space x ~ [-2, 2], y ~ [-2, 2], z ~ [2, 5]
    """
    pts = []
    for i in range(N):
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        z = np.random.uniform(40, 80)
        
        u, v, d = cam.inv_proj_uvd(x, y, z)
        pts.append([u, v, d])
    
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
    # pts_t = np.array([np.hstack((pn, pt)) for (pn, pt) in zip(pts_n, pts) if pn[0] > 0 and pn[0]< cam.width \
    #                 and pn[1] > 0 and pn[1] < cam.height and pn[2] > 0])
    pts_t = np.array([np.hstack((pn, pt)) for (pn, pt) in zip(pts_n, pts)])
    return pts_t[:,0:3].T, pts_t[:,3:].T
    

def get_pose():
    """
    generate different pose changes
    """

    angle = [3, 6, 9] # rotation [degrees]
    dis = [0.5, 1.0] # distance [m]
    
    pos = []
    
#    for roll in angle:
#        for pitch in angle:
#            for yaw in angle:
#                for x in dis:
#                    # for y in dis: 
#                        for z in dis:
#                          pos.append([x, 0., z, al.d2r(roll), al.d2r(pitch), al.d2r(yaw)])  
    # pos = [[1.5, 1.0, 0.7, al.d2r(9), al.d2r(9), al.d2r(9)]]
    pos = [[1., 0., 0., al.d2r(0), al.d2r(20), al.d2r(0)]]
    
#    pos.append([1, 0, 0, al.d2r(20), al.d2r(0), al.d2r(0)])
#    pos.append([0, 1, 0, al.d2r(20), al.d2r(0), al.d2r(0)])
#    pos.append([0, 0, 1, al.d2r(20), al.d2r(0), al.d2r(0)])
#
#    pos.append([1, 0, 0, al.d2r(0), al.d2r(20), al.d2r(0)])
#    pos.append([0, 1, 0, al.d2r(0), al.d2r(20), al.d2r(0)])
#    pos.append([0, 0, 1, al.d2r(0), al.d2r(20), al.d2r(0)])
#
#    pos.append([1, 0, 0, al.d2r(0), al.d2r(0), al.d2r(20)])
#    pos.append([0, 1, 0, al.d2r(0), al.d2r(0), al.d2r(20)])
#    pos.append([0, 0, 1, al.d2r(0), al.d2r(0), al.d2r(20)])    
    
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

def err_func(T, T_est):
    """
    Given gt T, and estimated T_est, compute relative distance and angular error 
    """
    eT = al.ominus(T, T_est)
    e_dis = al.compute_distance(eT)
    e_ang = al.compute_angle(eT)
    dis = al.compute_distance(T)
    ang = al.compute_angle(T)
    r_dis, r_ang = 0, 0
    if dis != 0:
        r_dis = e_dis / dis
    if ang != 0:
        r_ang = e_ang / ang 
    return r_dis, r_ang, e_dis, e_ang

def err_rot(R, R_est):
    """
    Given gt R, and estimated R_est, compute relative angular error 
    """
    eR = R.transpose().dot(R_est)
    ang = al.compute_angle(R)
    e_ang = al.compute_angle(eR)
    r_ang = 0.
    if ang != 0:
        r_ang = e_ang / ang 
    return  r_ang, e_ang
    

def compare_translation():
    """
    compare translation accuracy 3d-2d w/o 2d-2d for rotation n  
    """
    cam = cam_model.structCore()
    pos_set = get_pose()

    # npts = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    # npts = [30, 40, 60, 80, 120]
    npts = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] # 14, 16, 18, 20]
    # npts = [20, 30, 40]
    ea_2d2d = [] # mean angular error 
    ea_3d2d = [] 
    for npt in npts:
        ne_2d2d = []
        ne_3d2d = []
        for i in range(100):
            for pos in pos_set:
                
                 T = al.transform44Euler(pos)
                 R = T[0:3,0:3]
                 t = T[0:3,3]
                 print('gt: tij = {}, rpy = {}'.format(t.transpose(), al.r2d(al.rotationMatrixToEulerAngles(R).transpose())))
                 
                 R_inv = np.linalg.inv(R)
                 t_inv = -R_inv.dot(t)
                 # pts_i = get_pts_uvd(cam)
                 pts_i = get_pts(cam, noise = False)
                 pts_j, pts_i = transform_pts_uvd(R_inv, t_inv, cam, pts_i)
                 
                 
                 pts_j, pts_j_uv = pts_from_uvd(pts_j, cam, noise = True)
                 pts_i, pts_i_uv = pts_from_uvd(pts_i, cam, noise = True)
                 # Rij, tij, _ = al.align(np.asmatrix(pts_j), np.asmatrix(pts_i))
                 
                 Rij, _ = pnp_cv.pnp_2d2d_with_num(pts_j_uv[:2,:].transpose(), pts_i_uv[:2,:].transpose(), cam.K, n = 20)
                 # Rij, tij = pnp_cv.pnp_3d2d(pts_j.transpose(), pts_i_uv[:2,:].transpose(), cam.K)

                 t1, t2, _ = pnp_cv.pnp_3d2d_with_num_and_R(pts_j.transpose(), pts_i_uv[:2,:].transpose(), cam, Rij, n = npt)
                 
                 print('2d-2d est_t = {} 3d-2d est_t = {}'.format(t1, t2))
                 
                 rpy = al.rotationMatrixToEulerAngles(Rij)
                 e1 = err_rot(R, Rij)
                 
                 if np.max(np.abs(al.r2d(rpy))) > 40:
                     continue
                 
                 Rij, tij = pnp_cv.pnp_3d2d_with_num(pts_j.transpose(), pts_i_uv[:2,:].transpose(), cam.K, n = npt)
                 
                 e1 = np.linalg.norm(t1 - t)/np.linalg.norm(t)
                 
                 e2 = np.linalg.norm(t2.transpose() - t)/np.linalg.norm(t)
                 
                 if e1 > 0.6 or e2 > 0.6:
                     print('too large error result! ')
                     continue
                 
                 ne_2d2d.append(e1)
                 ne_3d2d.append(e2)
                 
        ne_2d2d = np.array(ne_2d2d)
        ea_2d2d.append(np.mean(ne_2d2d)) 
        
        ne_3d2d = np.array(ne_3d2d)
        ea_3d2d.append(np.mean(ne_3d2d))
        
    # draw the error 
    x = np.array(npts, dtype=np.int32)
    m_2d2d = np.array(ea_2d2d)*100.
    m_3d2d = np.array(ea_3d2d)*100.
    plt.close('all')
    fig, ax1 = plt.subplots(1,1)
    fig.subplots_adjust(hspace=.7)
    ax1.plot(x, m_2d2d,'b-o', label='2d-2d')
    ax1.set_ylabel('translation error [%]')
    ax1.plot(x, m_3d2d, 'r-s', label='3d-2d')
    
    # ax1.errorbar(x, m_ed, yerr = m_sd, fmt = 'ro')
    
    # ax1.plot(x, mean_da[:,1],'g-', label='')
    # ax1.plot(x, bias_a[:,2],'r-', label='bias_az')
    ax1.set_ylim([0,40])
    # plt.title('Accelerometer Bias')
    ax1.set_xlabel('Number of Inliers')
    ax1.set_title('Translation comparison 3d-2d vs 2d-2d')
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid()
    plt.savefig("./result/translation_comparison.png", dpi=360)
    
    plt.draw()


def compare_rotation():
    """
    compare rotation accuracy estimated by 3d-2d and 2d-2d
    """
    cam = cam_model.structCore()
    pos_set = get_pose()

    # npts = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    # npts = [30, 40, 60, 80, 120]
    npts = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
    # npts = [10, 14, 18, 20, 24, 28]
    ea_2d2d = [] # mean angular error 
    ea_3d2d = [] 
    for npt in npts:
        ne_2d2d = []
        ne_3d2d = []
        for i in range(100):
            for pos in pos_set:
                 T = al.transform44Euler(pos)
                 R = T[0:3,0:3]
                 t = T[0:3,3]
                 print('gt: tij = {}, rpy = {}'.format(t.transpose(), al.r2d(al.rotationMatrixToEulerAngles(R).transpose())))
                 # pts_j = get_pts_uvd(cam)
                 # pts_i, pts_j = transform_pts_uvd(R, t, cam, pts_j)
                 
                 R_inv = np.linalg.inv(R)
                 t_inv = -R_inv.dot(t)
                 # pts_i = get_pts_uvd(cam)
                 pts_i = get_pts(cam, noise = False)
                 pts_j, pts_i = transform_pts_uvd(R_inv, t_inv, cam, pts_i)
                 pts_j, pts_j_uv = pts_from_uvd(pts_j, cam, noise = True)
                 pts_i, pts_i_uv = pts_from_uvd(pts_i, cam, noise = True)
                 # Rij, tij, _ = al.align(np.asmatrix(pts_j), np.asmatrix(pts_i))
                 
                 Rij, tij = pnp_cv.pnp_2d2d_with_num(pts_j_uv[:2,:].transpose(), pts_i_uv[:2,:].transpose(), cam.K, n = npt)
                 # Rij, tij = pnp_cv.pnp_3d2d(pts_j.transpose(), pts_i_uv[:2,:].transpose(), cam.K)
                 rpy = al.rotationMatrixToEulerAngles(Rij)
                 print('2d-2d est: rpy = ', al.r2d(rpy))
                 e1 = err_rot(R, Rij)
                 
                 if np.max(np.abs(al.r2d(rpy))) > 40:
                     continue
                 
                 Rij, tij = pnp_cv.pnp_3d2d_with_num(pts_j.transpose(), pts_i_uv[:2,:].transpose(), cam.K, n = npt)
                  
                 rpy = al.rotationMatrixToEulerAngles(Rij)
                 print('3d-2d est: rpy = ', al.r2d(rpy)) 
                 
                 if np.max(np.abs(al.r2d(rpy))) > 40:
                     continue
                 
                 e2 = err_rot(R, Rij)
                 if e1[0] > 0.6 or e2[0] > 0.6:
                     print('the result is not right!')
                     continue
                 ne_2d2d.append(e1)
                 ne_3d2d.append(e2)
                 
        ne_2d2d = np.array(ne_2d2d)[:,0]
        ea_2d2d.append(np.mean(ne_2d2d)) 
        
        ne_3d2d = np.array(ne_3d2d)[:,0]
        ea_3d2d.append(np.mean(ne_3d2d))
        
    # draw the error 
    x = np.array(npts, dtype=np.int32)
    m_2d2d = np.array(ea_2d2d)*100.
    m_3d2d = np.array(ea_3d2d)*100.
    plt.close('all')
    fig, ax1 = plt.subplots(1,1)
    fig.subplots_adjust(hspace=.7)
    ax1.plot(x, m_2d2d,'b-o', label='2d-2d')
    ax1.set_ylabel('rotation error [%]')
    ax1.plot(x, m_3d2d, 'r-s', label='3d-2d')
    
    # ax1.errorbar(x, m_ed, yerr = m_sd, fmt = 'ro')
    
    # ax1.plot(x, mean_da[:,1],'g-', label='')
    # ax1.plot(x, bias_a[:,2],'r-', label='bias_az')
    ax1.set_ylim([0,20])
    # plt.title('Accelerometer Bias')
    ax1.set_xlabel('Number of Inlier1s')
    ax1.set_title('Rotation comparison 3d-2d vs 2d-2d')
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid()
    plt.savefig("./result/rotation_comparison.png", dpi=360)
    
    plt.draw()
    

def generate_data_for_test():
    cam = cam_model.structCore()
    pos_set = get_pose()

    # npts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # npts = [7, 14, 21, 28, 35, 42, 49]
    # npts = [6, 9, 12, 15, 18, 21, 24, 27, 30]
    # npts = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    npts = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    # npts = [20, 30, 40]
    m_ed = [] # mean distance error 
    m_ea = [] # mean angular error 
    m_sd = [] # std distance error 
    m_sa = [] # std angular error 
    for npt in npts:
        ne = []
        for i in range(100):
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
                 
                 # Rij, tij = pnp_cv.pnp_2d2d_num(pts_j_uv[:2,:].transpose(), pts_i_uv[:2,:].transpose(), cam.K)
                 # Rij, tij = pnp_cv.pnp_3d2d(pts_j.transpose(), pts_i_uv[:2,:].transpose(), cam.K)
    
                 Rij, tij = pnp_cv.pnp_3d2d_with_num(pts_j.transpose(), pts_i_uv[:2,:].transpose(), cam.K, n = npt)
                  
                 rpy = al.rotationMatrixToEulerAngles(Rij)
                 print('est: tij = {}, rpy = {}'.format(tij.transpose(), al.r2d(rpy))) 
                 
                 # compute error 
                 T_est = al.transformRt(Rij, tij)
                 e = err_func(T, T_est)
                 if e[0] > 0.6 or e[1] > 0.6:
                     continue
                 ne.append(e)
                 print('error is : ', e)
                 
                 
        ne = np.array(ne)[:,0:2]
        mean_da = np.mean(ne, axis = 0)
        std_da  = np.std(ne, axis = 0)
        m_ed.append(mean_da[0]) 
        m_ea.append(mean_da[1])
        m_sd.append(std_da[0])
        m_sa.append(std_da[1])
        
    # draw the error 
    x = np.array(npts, dtype=np.int32)
    m_ed = np.array(m_ed)*100.
    m_ea = np.array(m_ea)*100.
    m_sd = np.array(m_sd)*0. # not shown standard
    m_sa = np.array(m_sa)*0.
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2,1)
    fig.subplots_adjust(hspace=.7)
    ax1.plot(x, m_ed,'b-o', label='distance error percentage [%]')
    ax1.set_ylabel('translation error [%]')
    # ax1.errorbar(x, m_ed, yerr = m_sd, fmt = 'ro')
    
    # ax1.plot(x, mean_da[:,1],'g-', label='')
    # ax1.plot(x, bias_a[:,2],'r-', label='bias_az')
    ax1.set_ylim([0,20])
    # plt.title('Accelerometer Bias')
    ax1.set_xlabel('Number of Inliers')
    # ax1.set_title('Distance error in Percentage [%]')
    # ax1.legend(loc='upper right', fontsize='small')
    # plt.savefig("../intermidiate_files/"+data_name+"_bias_acc.png", dpi=360)
    
    # plt.subplot(2,1,2)
    ax2.plot(x, m_ea,'b-o', label='angular error percentage [%]')
    # ax2.errorbar(x, m_ea, yerr = m_sa, fmt = 'ro')
    
    # plt.title('Accelerometer Bias')
    ax2.set_ylim([0,20])
    ax2.set_xlabel('Number of Inliers')
    # ax2.set_title('Angular error Percentage [%]')
    ax2.set_ylabel('angular error percentage')
    # ax2.legend(loc='upper right', fontsize='small')
    plt.savefig("./result/3d-2d.png", dpi=360)   
    plt.draw()

def test_one_pose():
    cam = cam_model.structCore()
    
    pts_j = get_pts_uvd(cam)
    Pij = [0.2, 0., 0., al.d2r(0.), al.d2r(20.), al.d2r(0.)]
    # Pij = [0., 0., 0., al.d2r(0.), al.d2r(0.), al.d2r(0.)]
    T = al.transform44Euler(Pij)
    R = T[0:3,0:3]
    t = T[0:3,3]
    pts_i, pts_j = transform_pts_uvd(R, t, cam, pts_j)
    pts_j, pts_j_uv = pts_from_uvd(pts_j, cam, noise = True)
    pts_i, pts_i_uv = pts_from_uvd(pts_i, cam, noise = True)
    # Rij, tij, _ = al.align(np.asmatrix(pts_j), np.asmatrix(pts_i))
    
    # Rij, tij = pnp_cv.pnp_2d2d(pts_j_uv[:2,:].transpose(), pts_i_uv[:2,:].transpose(), cam.K)
    
    # Rij, tij = pnp_cv.pnp_2d2d_with_num(pts_j_uv[:2,:].transpose(), pts_i_uv[:2,:].transpose(), cam.K, n = 50)
    
    Rij, tij = pnp_cv.pnp_3d2d_with_num(pts_j.transpose(), pts_i_uv[:2,:].transpose(), cam.K, n = 50)
    # Rij, tij = pnp_cv.pnp_3d2d(pts_j.transpose(), pts_i_uv[:2,:].transpose(), cam.K)
    
    # Rij, tij = pnp_cv.pnp_3d3d_with_num(pts_j.transpose(), pts_i_uv[:2,:].transpose(), pts_i.transpose(), cam.K, n = 50)
    # Rij  = R
    # tij = lsq_tR.lsq_solve(pts_j.transpose(), pts_i_uv[:2,:].transpose(), Rij, cam)
    rpy = al.rotationMatrixToEulerAngles(Rij)
    print('tij = {}, rpy = {}'.format(tij.transpose(), al.r2d(rpy)))
    
    # computer error 
    T_est = al.transformRt(Rij, tij)
    e = err_func(T, T_est)
    print('error is : ', e)
    
def debug():
    pass

if __name__=="__main__":
    compare_translation()
    compare_rotation()
    # generate_data_for_test()
    # test_one_pose()

    
    
    
    
    
    
    
    
    


        


