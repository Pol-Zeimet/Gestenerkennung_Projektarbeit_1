import numpy as np               # fundamental package for scientific computing
import pyrealsense2 as rs        # Intel RealSense cross-platform open-source API
print("Environment Ready")
filename = ""
# Create a context object. This object owns the handles to all connected realsense devices
try:
    config = rs.config()
    rs.config.enable_device_from_file(config, filename, repeat_playback=False)
    pipe = rs.pipeline()
    profile = pipe.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    while True:
        frames = pipe.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth: continue

        # Print a simple text-based representation of the image, by breaking it into 10x20 pixel regions and approximating the coverage of pixels within one meter
        coverage = [0] * 64
        for y in range(480):
            for x in range(640):
                dist = depth.get_distance(x, y)
                if 0 < dist and dist < 1:
                    coverage[x // 10] += 1

            if y % 20 is 19:
                line = ""
                for c in coverage:
                    line += " .:nhBXWW"[c // 25]
                coverage = [0] * 64
                print(line)
except Exception as e:
    print(e)
    pass
