# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:40:40 2019

@author: fuyin

camera models

"""


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
    
    def project(self, u, v, z):
        x = z*(u-self.cx)/self.fx
        y = z*(v-self.cy)/self.fy
        return x,y,z

class structCore(camModel):
    """
    AR core camera model 
    """
    def __init__(self):
        camModel.__init__(self, 556.875, 556.875, 295.5, 232.25, 640, 480)
        self.l = 0.1
        