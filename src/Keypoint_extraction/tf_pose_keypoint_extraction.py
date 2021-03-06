#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:30:47 2019

@author: Pol Zeimet
"""

import argparse
import logging
import time

import numpy as np
import pandas as pd

import pyrealsense2 as rs
import cv2

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh



#using tensorflow <  2.0

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
align_to = rs.stream.color
align = rs.align(align_to)
frames = pd.DataFrame()
depth_intrins = None


def getKeyPointCoords(keypoints_in, depth_frame, depth_colormap):
    keyPoints_out = []
    image_h, image_w = depth_colormap.shape[:2]
    base_keypoint = None
    if 8 in keypoints_in.keys() and 11 in keypoints_in.keys():
        base_x = int((keypoints_in[8][0] + keypoints_in[11][0])/2)
        base_y = int((keypoints_in[8][1] + keypoints_in[11][1])/2)
        base_z = (depth_frame.get_distance(keypoints_in[8][0], keypoints_in[8][1]) + depth_frame.get_distance(keypoints_in[11][0], keypoints_in[11][1]))/2.
        cv2.circle(depth_colormap, (base_x, base_y), 2, common.CocoColors[10], thickness=2, lineType=8)
        coords = transformPixelToCameraCoords([base_x, base_y, base_z])
        base_keypoint = [coords[0], coords[1], coords[2]]
        keyPoints_out.append({'Keypoint': 'base_keypoint','X_pix_pos': base_x, 'Y_pix_pos' : base_y, 'Z_pix_pos' : base_z, 'X_cam_pos': coords[0], 'Y_cam_pos' : coords[1], 'Z_cam_pos' : coords[2], 'X_body_pos' : None, 'Y_body_pos' : None, 'Z_body_pos' : None})
    else:
        keyPoints_out.append({'Keypoint': 'base_keypoint','X_pix_pos': None, 'Y_pix_pos' : None, 'Z_pix_pos' : None, 'X_cam_pos': None, 'Y_cam_pos' : None, 'Z_cam_pos' : None, 'X_body_pos' : None, 'Y_body_pos' : None, 'Z_body_pos' : None})
    
    for i in range(common.CocoPart.Background.value):
        if i not in keypoints_in.keys():
            keyPoints_out.append({'Keypoint': i, 'X_pix_pos': None, 'Y_pix_pos' : None, 'Z_pix_pos' : None, 'X_cam_pos': None, 'Y_cam_pos' : None, 'Z_cam_pos' : None, 'X_body_pos' : None, 'Y_body_pos' : None, 'Z_body_pos' : None})
            continue
        z_pos = depth_frame.get_distance(keypoints_in[i][0], keypoints_in[i][1])
        coords = transformPixelToCameraCoords([keypoints_in[i][0], keypoints_in[i][1], z_pos])
        keyPoints_out.append({'Keypoint': i, 'X_pix_pos': keypoints_in[i][0], 'Y_pix_pos' : keypoints_in[i][1], 'Z_pix_pos' : z_pos, 'X_cam_pos': coords[0], 'Y_cam_pos' : coords[1], 'Z_cam_pos' : coords[2], 'X_body_pos' : None, 'Y_body_pos' : None, 'Z_body_pos' : None})
 
    if base_keypoint:
        rotation_matrix = np.column_stack(getBodyBaseVectors(keyPoints_out))
        for keypoint in keyPoints_out:
            if keypoint['X_cam_pos'] != None:
                coords = transformCameraToBodyCoords(rotation_matrix, base_keypoint, [keypoint['X_cam_pos'], keypoint['Y_cam_pos'], keypoint['Z_cam_pos']])
                keypoint['X_body_pos'] = coords[0]
                keypoint['Y_body_pos'] = coords[1]
                keypoint['Z_body_pos'] = coords[2]
    
    return keyPoints_out, depth_colormap

def transformPixelToCameraCoords(coords): # Pixel to Camera Coordinates
    return rs.rs2_deproject_pixel_to_point(depth_intrins, [coords[0], coords[1]], coords[2])


def transformCameraToBodyCoords(rotation_matrix, base_keypoint, coord):
    rc = np.array(coord)
    hc = np.array(base_keypoint)
    vector = rotation_matrix.dot(rc - hc)
    return vector


def getVector(coord_1, coord_2):
    a = np.array(coord_1)
    b = np.array(coord_2)
    return b - a

def getBodyBaseVectors(keypoints):
    base_vector_x_norm = [0, 0, 0]
    base_vector_y_norm = [0, 0, 0]
    if keypoints[2]['X_cam_pos'] != None and  keypoints[12]['X_cam_pos'] != None:
        base_vector_x = getVector([keypoints[0]['X_cam_pos'],keypoints[0]['Y_cam_pos'], keypoints[0]['Z_cam_pos']], [keypoints[12]['X_cam_pos'],keypoints[12]['Y_cam_pos'], keypoints[12]['Z_cam_pos']])
        base_vector_y = getVector([keypoints[0]['X_cam_pos'],keypoints[0]['Y_cam_pos'], keypoints[0]['Z_cam_pos']], [keypoints[2]['X_cam_pos'],keypoints[2]['Y_cam_pos'], keypoints[2]['Z_cam_pos']])
       
        if np.linalg.norm(base_vector_x) != 0:
            base_vector_x_norm = base_vector_x / np.linalg.norm(base_vector_x)
                
        if np.linalg.norm(base_vector_y) != 0:
            base_vector_y_norm = base_vector_y / np.linalg.norm(base_vector_y)

    base_vector_z_norm = np.cross(base_vector_x_norm, base_vector_y_norm)
    
    return base_vector_x_norm, base_vector_y_norm, base_vector_z_norm

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

   
def draw_human_v2(npimg, human, imgcopy=False):
        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        centers = {}
        
        # draw point
        for i in range(common.CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue
            body_part = human.body_parts[i]
            center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            centers[i] = center
            cv2.circle(npimg, center, 2, common.CocoColors[i], thickness=2, lineType=8, shift=0)
            #new: draw Keypoint number
            #cv2.putText(npimg, text = str(i), org = (int(body_part.x * image_w + 0.25),int(body_part.y * image_h + 0.25)), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale =  1, color = (255, 255, 255))
        # draw line
        for pair_order, pair in enumerate(common.CocoPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue
            # npimg = cv2.line(npimg, cegenererateBasepointnters[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
            cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 1)

        return npimg, centers





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')

    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=1.5,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_v2_small', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="True",
                        help='for tensorrt process.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator('../../models/custom_graph.pb', target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(368, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    
    pipeline.start(config)
    while True:
        frame = pipeline.wait_for_frames()
        aligned_frames = align.process(frame)
        depth_frame = aligned_frames.get_depth_frame()

        # calibriated internal parameters from RealSense
        depth_intrins = depth_frame.profile.as_video_stream_profile().intrinsics

        color_frame = aligned_frames.get_color_frame()
        
        if not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.2), cv2.COLORMAP_JET)
        
        
        logger.debug('image process+')
        humans = e.inference(color_image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        
        logger.debug('postprocess+')
        if len(humans) > 0:
            human = humans[0]
            color_image, keypoints = draw_human_v2(color_image, human, imgcopy=False)
            coords_pixel, depth_colormap = getKeyPointCoords(keypoints, depth_frame, depth_colormap)
            frames = frames.append(coords_pixel)
            
        logger.debug('show+')
        cv2.putText(color_image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        images = np.hstack((color_image,  depth_colormap))
        cv2.imshow('tf-pose-estimation result', images)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')
        
    cv2.destroyAllWindows()
    print(frames)
    frames.to_csv('../data/keypoints_test.csv', encoding = 'UTF-8', sep= ';')
