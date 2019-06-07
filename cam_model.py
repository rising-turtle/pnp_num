# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:40:40 2019

@author: fuyin

camera models

"""
import numpy as np 

class camModel:
    """
    camera model 
    """
    def __init__(self, fx, fy, cx, cy, w, h):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = w
        self.height = h
        self.K = np.array(
                        [[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]], dtype = "double"
                         )
    
    def norm_pt(self, u, v):
        return (u-self.cx)/self.fx, (v-self.cy)/self.fy
    
    def project(self, u, v, z):
        x = z*(u-self.cx)/self.fx
        y = z*(v-self.cy)/self.fy
        return x,y,z
    
    def inv_proj(self, x, y, z):
        u = x*self.fx/z + self.cx
        v = y*self.fy/z + self.cy
        return u,v

class structCore(camModel):
    """
    AR core camera model 
    """
    def __init__(self):
        camModel.__init__(self, 556.875, 556.875, 295.5, 232.25, 640, 480)
        self.l = 0.075 # 75mm
    def project_uvd(self, u, v, d):
        z = self.l*self.fx/d 
        return camModel.project(self, u, v, z)
    def inv_proj_uvd(self, x, y, z):
        u,v = camModel.inv_proj(self, x, y, z)
        d = self.l*self.fx/z
        return u, v, d

if __name__=="__main__":
    
    cam = structCore()
    
    u, v, d = 10, 10, 8.5
    x, y, z = cam.project_uvd(u, v, d)
    print('u, v, d = ', u, v, d)
    print('x, y, z = ', x, y, z)
    u, v, d = cam.inv_proj_uvd(x, y, z)
    print('u, v, d = ', u, v, d)
    
#    u, v, z = 10, 10, 5.89
#    x,y,z = -2.42, -2.24, 5.89
#    # u, v = cam.inv_proj(x, y, z)
#    print('x, y, z = ', x, y, z)
#    print('u, v, z = ', u, v, z)
#    x, y, z = cam.project(u, v, z)
#    u, v = cam.inv_proj(x, y, z)
#    print('x, y, z = ', x, y, z)
#    print('u, v, z = ', u, v, z)
    
    
    
    
    
    