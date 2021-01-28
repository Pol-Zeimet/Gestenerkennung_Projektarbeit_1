# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:25:47 2019

@author: Pol
"""
"""
execute 
$ PYTHONPATH=$PYTHONPATH:/usr/local/lib
in cmd first
"""

# First import the library
import pyrealsense2 as rs
import os
import time

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
config = rs.config()

timeStr = time.strftime("%Y%m%d-%H%M%S")

filename = 'recording_' + timeStr.replace('-', '_') + '.bag'
print(filename)
config.enable_record_to_file(filename)
pipe_running = False
try:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    align_to = rs.stream.color
    align = rs.align(align_to)
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline.start(config)
    pipe_running = True
    i = 0
    while i <= 400:
        i+=1
        # This call waits until a new coherent set of frames is available on a device
        # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        frame = pipeline.wait_for_frames()
        print('frame', i)

except Exception as e:
    print(e)
    pass
finally:
    if pipe_running:
        pipeline.stop()
    pipeline = None
    config = None
    print('stopped pipeline')

print('moving file to destination')
os.rename(filename, "../../data/recording_samples/bag_files/"+filename)

