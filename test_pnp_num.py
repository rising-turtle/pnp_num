# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:32:58 2019

@author: fuyin

find out the relationship between the accuracy of pnp and the number of points 

"""

import cam_model
import align
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
    pts = np.array([[x, y, z] for (x,y,z) in zip(px, py, pz)])
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


if __name__=="__main__":
    
    cam = cam_model.structCore()
    pts = get_pts(cam)
    
    
    
    
    
    
    
    
    


        


