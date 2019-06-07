# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:12:53 2019

lsq my own problems 

@author: fuyin
"""

import numpy as np
from scipy.optimize import least_squares
import cam_model

def residual_fun(t, x_train, y_train):
    """
    r1 = y1 + (tx - X*tz) 
    r2 = y2 + (ty - Y*tz)
    where y1 = (R1 - X*R3)*x
    y2 = (R2 - Y*R3)*x
    """
    e = np.array([ [y[0]+t[0] - x[0]*t[2], y[1] + t[1] - x[1]*t[2]] for (x,y) in zip(x_train, y_train)])
    return e.reshape(-1)

def y_trans(R, cam, model_point, image_point):
    """
    compute y = (R1 - X*R3)*x, (R2 - Y*R3)*x
    
    input:
        image_point [nx2]
        mpdel_point [nx3]
    output:
        y [n,2]: 
        x_norm [n,2]
    """
    R1 = R[0, :]
    R2 = R[1, :]
    R3 = R[2, :]
    
    x_norm = np.array([ cam.norm_pt(img_pt[0], img_pt[1]) for img_pt in image_point])
    
    y = np.array([ [(R1 - pt2[0]*R3).dot(pt3.transpose()), (R2- pt2[1]*R3).dot(pt3.transpose())] 
            for (pt3, pt2) in zip(model_point, x_norm)])
    
    return y, x_norm


def lsq_solve(model_point, image_point, R, cam, t0 = [0.0, 0.0, 0.0]):
    """
    given R, solve t, using leqst square method 
    
    input:
        model_point [nx3]
        image_point [nx2]
        R [3x3]
        cam: camera model 
    output:
        t [3,]
    """
    y_train, t_train = y_trans(R, cam, model_point, image_point)
    
    #%% least square fit 
    x0 = np.array(t0)
    
    # r = residual_fun(x0, t_train, y_train)
    
    res_lsq = least_squares(residual_fun, x0, args = (t_train, y_train))
    print('lsq res.x = {} cost = {} optimality = {}'.format(res_lsq.x, res_lsq.cost, res_lsq.optimality))
    return res_lsq.x

if __name__ == "__main__":
    
    cam = cam_model.structCore()
    
    R = np.identity(3)
    t = np.array([1.0, 2.0, 0.5])
    
    pts_j = np.array([[np.random.uniform(-2,2), np.random.uniform(-2,2), np.random.uniform(2,7)] for i in range(10)])
    
    pts_i = np.array([(R.dot(pt.transpose()) + t.transpose()).transpose() for pt in pts_j])
    img_pt = np.array([(pt[0]*cam.fx/pt[2] + cam.cx, pt[1]*cam.fy/pt[2] + cam.cy) for pt in pts_i])
    
    t_est = lsq_solve(pts_j, img_pt, R, cam)
    
    
    
    
    
    
    
    
