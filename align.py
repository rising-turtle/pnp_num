# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:37:01 2019

@author: fuyin

compute transformation matrix given two point clouds 

"""

import numpy as np
import math

_EPS = np.finfo(float).eps * 4.0

def d2r(d):
    return d*math.pi/180.

def r2d(r):
    return r*180./math.pi

 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z

def align(model,data):
    """Align two point clouds using the method of Horn (closed-form).
    
    Input:
    model -- first point cloud (3xn)
    data -- second point cloud (3xn)
    
    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    
    error = rot * model + trans - data
    
    """
    np.set_printoptions(precision=3,suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)
    
    W = np.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity( 3 ))
    if(np.linalg.det(U) * np.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh
    trans = data.mean(1) - rot * model.mean(1)
    
    model_aligned = rot * model + trans
    alignment_error = model_aligned - data
    
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]
        
    return rot,trans,trans_error

def transformRt(R, t):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and Rotation matrix.
    
    Input:
    R -- Rotation matrix
    t -- (tx,ty,tz) is the 3D position 
         
    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    return np.array((
    (            R[0][0],          R[0][1],         R[0][2], t[0]),
    (            R[1][0],          R[1][1],         R[1][2], t[1]),
    (            R[2][0],          R[2][1],         R[2][2], t[2]),
    (                0.0,                 0.0,                 0.0, 1.0)
    ), dtype=np.float64)

def transform44Euler(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion euler angles.
    
    Input:
    l -- tuple consisting of (tx,ty,tz,roll, pitch, yaw) where
         (tx,ty,tz) is the 3D position and (roll, pitch, yaw) is the euler angle.
         
    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    # q = np.zeros((1,7))
    q = euler_to_quaternion(l[3], l[4], l[5])
    a = (l[0], l[1], l[2], q[0], q[1], q[2], q[3])
    
    return transform44(a)
    

def transform44(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.
    
    Input:
    l -- tuple consisting of (tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.
         
    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[0:3]
    q = np.array(l[3:7], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.array((
        (                1.0,                 0.0,                 0.0, t[0])
        (                0.0,                 1.0,                 0.0, t[1])
        (                0.0,                 0.0,                 1.0, t[2])
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], t[0]),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], t[1]),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], t[2]),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

def rotation_matrix(l):
    """
    generate rotation matrix from quaternion
    """
    q = np.array(l[:], dtype=np.float64, copy = True)
    nq = np.dot(q,q)
    if nq < _EPS:
        return np.array((
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0)
                        ), dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
            (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3]),
            (q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3]),
            (q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1])
            ), dtype=np.float64)
    

def ominus(a,b):
    """
    Compute the relative 3D transformation between a and b.
    
    Input:
    a -- first pose (homogeneous 4x4 matrix)
    b -- second pose (homogeneous 4x4 matrix)
    
    Output:
    Relative 3D transformation from a to b.
    """
    return np.dot(np.linalg.inv(a),b)

def compute_distance(transform):
    """
    Compute the distance of the translational component of a 4x4 homogeneous matrix.
    """
    return np.linalg.norm(transform[0:3,3])

def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    # an invitation to 3-d vision, p 27
    return np.arccos( min(1,max(-1, (np.trace(transform[0:3,0:3]) - 1)/2) ))

if __name__ == "__main__":
    
    r, p, y = d2r(10), d2r(20), d2r(30)
    q = euler_to_quaternion(r, p, y)
    r, p, y = quaternion_to_euler(q[0], q[1], q[2], q[3])
    print("q = {}".format(q))
    print("r, q, y = {},{},{}".format((r), (p), (y)))
    
    R = rotation_matrix(q)
    t = np.array([1.0, 2.0, 3.0])
    l = np.hstack((t, q))
    T1 = transform44(l)
    T2 = transformRt(R,t)
    print("T1 = \n{}".format(T1))
    print("T2 = \n{}".format(T2))
    
    
    
    
    
    