# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:13:55 2019

@author: pol
"""


"""
execute 
$ PYTHONPATH=$PYTHONPATH:/usr/local/lib
in cmd first
"""

# From Python
# It requires OpenCV installed for Python
import cv2
import pyrealsense2 as rs
import numpy as np



# Read Webcam
pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)
while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
            
    if not depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.2), cv2.COLORMAP_JET)

    cv2.putText(color_image, text = 'test', org = (100,100),fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale =  5, color = (255, 255, 255))
    images = np.hstack((color_image, depth_colormap))
    cv2.imshow("OpenCV, Realsense Test", images)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


