# -*- coding: utf-8 -*-
"""
Created on Thu May 30 06:08:49 2019

@author: fuyin

pnp method 

"""

import cv2 
import numpy as np 
import cam_model 
import lsq_t_with_R as ltr
import align as al

def pnp_2d2d_with_num(pts_j, pts_i, K, n):
    """
    pnp given the number of inliers 
    """
    F, mask = cv2.findFundamentalMat(pts_j.astype("double"),pts_i.astype("double"),cv2.FM_RANSAC)
    
    # find inliers 
    ptsj = pts_j[mask.ravel()==1]
    ptsi = pts_i[mask.ravel()==1]
    
    if ptsj.shape[0] < n:
        print('2d-2d few inliers number {} < N = {}'.format( ptsj.shape[0], n))
        return None, None
    
    s = np.int32(ptsj.shape[0] / n) 
    ptsi = ptsi[::s]
    ptsj = ptsj[::s]
    
    F, _ = cv2.findFundamentalMat(ptsj, ptsi, cv2.FM_8POINT)
    
    E = np.matmul(np.matmul(K.T, F), K)
    
    points, R, t, mask = cv2.recoverPose(E, ptsj, ptsi)
    
    return R, t

def pnp_2d2d(pts_j, pts_i, K):
    # Normalize for Esential Matrix calaculation
    
    # print('src.rows = {} src.isContinuous {} src.depth() = {} src.channels() = {} src.cols = {}', pts_i.rows, \
    #      pts_i.isContinuous, pts_i.depth(), pts_i.channels(), pts_i.cols)
    pts_j = pts_j.astype("double")
    pts_i = pts_i.astype("double")
    # pts_l_norm = cv2.undistortPoints(np.expand_dims(pts_j, axis=1), cameraMatrix=K, distCoeffs=None)
    # pts_l_norm = cv2.undistortPoints(pts_j, cameraMatrix=K, distCoeffs=None)
    # pts_r_norm = cv2.undistortPoints(np.expand_dims(pts_i, axis=1), cameraMatrix=K, distCoeffs=None)
    # E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    F, mask = cv2.findFundamentalMat(pts_j,pts_i,cv2.FM_RANSAC)
    
    E = np.matmul(np.matmul(K.T, F), K)
    
    points, R, t, mask = cv2.recoverPose(E, pts_j, pts_i)
    
    return R, t
    
    # M_r = np.hstack((R, t))
    #M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

    # P_l = np.dot(K_l,  M_l)
    # P_r = np.dot(K_r,  M_r)
    # point_4d_hom = cv2.triangulatePoints(P_l, P_r, np.expand_dims(pts_l, axis=1), np.expand_dims(pts_r, axis=1))
    # point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    # point_3d = point_4d[:3, :].T

def pnp_3d3d_with_num(model_points, image_points, image_pts_3d, K, n, dists = np.zeros((4,1))):
    _, rotation_vector, translation_vector, mask = cv2.solvePnPRansac(model_points, image_points, K, dists)
    
    pt3d = model_points[mask.ravel()]
    pt2d = image_points[mask.ravel()]
    pt3d_img = image_pts_3d[mask.ravel()]
    
    if pt3d.shape[0] < n:
        print('3d-2d few inliers number {} < N = {}'.format( pt3d.shape[0], n))
        return None, None
    
    s = np.int32(pt3d.shape[0] / n) 
    pt3d = pt3d[::s]
    pt2d = pt2d[::s]
    pt3d_img = pt3d_img[::s]
    
    R, t, _ = al.align(np.asmatrix(pt3d.transpose()), np.asmatrix(pt3d_img.transpose()))
    return np.array(R), np.array(t)
    

def pnp_3d2d_with_num_and_R(model_points, image_points, cam, R, n, dists = np.zeros((4,1))):
    """
    given rotation R, select n points to compute translation 
    """
    K = cam.K
    _, rotation_vector, translation_vector, mask = cv2.solvePnPRansac(model_points, image_points, K, dists)
    
    pt3d = model_points[mask.ravel()]
    pt2d = image_points[mask.ravel()]
    
    if pt3d.shape[0] < n:
        print('3d-2d few inliers number {} < N = {}'.format( pt3d.shape[0], n))
        return None, None
    
    s = np.int32(pt3d.shape[0] / n) 
    pt3d = pt3d[::s]
    pt2d = pt2d[::s]
    
    _, rotation_vector, translation_vector =  cv2.solvePnP(pt3d.astype("double"), pt2d.astype("double"), K, dists)
    R2 = cv2.Rodrigues(rotation_vector)[0]
    t = translation_vector
    
    #% compute t using R  
    
    t2 = ltr.lsq_solve(pt3d, pt2d, R, cam)
    
    return t2, t, R2  
    

def pnp_3d2d_with_num(model_points, image_points, K, n, dists = np.zeros((4,1))):
    _, rotation_vector, translation_vector, mask = cv2.solvePnPRansac(model_points, image_points, K, dists)
    
    pt3d = model_points[mask.ravel()]
    pt2d = image_points[mask.ravel()]
    
    if pt3d.shape[0] < n:
        print('3d-2d few inliers number {} < N = {}'.format( pt3d.shape[0], n))
        return None, None
    
    s = np.int32(pt3d.shape[0] / n) 
    pt3d = pt3d[::s]
    pt2d = pt2d[::s]
    
    # pt3d = 
    N = pt2d.shape[0]
    pt2d = np.ascontiguousarray(pt2d[:,:2]).reshape((N,1,2))
    _, rotation_vector, translation_vector = cv2.solvePnP(pt3d.astype("double"), pt2d.astype("double"), K, dists, flags=cv2.SOLVEPNP_EPNP)
     
    # _, rotation_vector, translation_vector =  cv2.solvePnP(pt3d.astype("double"), pt2d.astype("double"), K, dists)
    R = cv2.Rodrigues(rotation_vector)[0]
    t = translation_vector
    return R, t    


def pnp_3d2d(model_points, image_points, K, dists = np.zeros((4,1))):
    # Normalize for Esential Matrix calaculation
    model_points = model_points.astype("double")
    image_points = image_points.astype("double")
    _, rotation_vector, translation_vector, mask = cv2.solvePnPRansac(model_points, image_points, K, dists)
    
    pt3d = model_points[mask.ravel()]
    pt2d = image_points[mask.ravel()]
    # _, rotation_vector, translation_vector = cv2.solvePnP(pt3d.astype("double"), pt2d.astype("double"), K, dists, flags=cv2.SOLVEPNP_EPNP)
    
    _, rotation_vector, translation_vector = cv2.solvePnP(pt3d.astype("double"), pt2d.astype("double"), K, dists)
    
    R = cv2.Rodrigues(rotation_vector)[0]
    t = translation_vector
    return R, t    
    
def example():
 
    # Read Image
    im = cv2.imread("headPose.jpg");
    size = im.shape
         
    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                                (359, 391),     # Nose tip
                                (399, 561),     # Chin
                                (337, 297),     # Left eye left corner
                                (513, 301),     # Right eye right corne
                                (345, 465),     # Left Mouth corner
                                (453, 469)      # Right mouth corner
                            ], dtype="double")
     
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                             
                            ])
     
     
    # Camera internals
     
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
     
    print ("Camera Matrix :\n {0}".format(camera_matrix))
     
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    # (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)
    
    _, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(model_points, image_points, camera_matrix, dist_coeffs)
    
    print ("Rotation Vector:\n {0}".format(rotation_vector))
    print ("Translation Vector:\n {0}".format(translation_vector))
    
    R = cv2.Rodrigues(rotation_vector)[0]
    print ("rotation matrix: \n {}".format(R))
    
    R, t = pnp_3d2d(model_points, image_points, camera_matrix)
    print ("rotation matrix: \n {}".format(R))
    print ("Translation vector: \n {}".format(t))
     
    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
     
     
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
     
    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
     
     
    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
     
    cv2.line(im, p1, p2, (255,0,0), 2)
     
    # Display image
    # cv2.imshow("Output", im)
    # cv2.waitKey(0)    

if __name__=="__main__":
    example()
    
