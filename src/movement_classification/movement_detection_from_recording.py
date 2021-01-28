import sys
sys.path.append("../../thirdparty/tf-pose/tf-pose-estimation")
import argparse
import logging
import traceback
import time
import numpy as np
import pyrealsense2 as rs
import cv2
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import tensorflow as tf
print(tf.__version__)
print("Environment Ready")


def getKeyPointCoords(keypoints_in, depth_frame, depth_colormap):
    keypoints_cam = []
    keypoints_out = []
    image_h, image_w = depth_colormap.shape[:2]
    base_keypoint = None
    if 8 in keypoints_in.keys() and 11 in keypoints_in.keys():
        base_x = int((keypoints_in[8][0] + keypoints_in[11][0]) / 2)
        base_y = int((keypoints_in[8][1] + keypoints_in[11][1]) / 2)
        base_z = (depth_frame.get_distance(keypoints_in[8][0], keypoints_in[8][1]) + depth_frame.get_distance(
            keypoints_in[11][0], keypoints_in[11][1])) / 2.
        cv2.circle(depth_colormap, (base_x, base_y), 2, common.CocoColors[10], thickness=2, lineType=8)
        coords = transformPixelToCameraCoords([base_x, base_y, base_z])
        base_keypoint = [coords[0], coords[1], coords[2]]
        keypoints_cam.append(base_keypoint)

    if base_keypoint:
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 11]:
            if i in keypoints_in.keys():
                z_pos = depth_frame.get_distance(keypoints_in[i][0], keypoints_in[i][1])
                coords = transformPixelToCameraCoords([keypoints_in[i][0], keypoints_in[i][1], z_pos])
                keypoints_cam.append(coords)
            else:
                return None, depth_colormap
        rotation_matrix = np.column_stack(getBodyBaseVectors(keypoints_cam))
        for keypoint in keypoints_cam:
            keypoints_out.append(transformCameraToBodyCoords(rotation_matrix, base_keypoint, keypoint))
    else:
        return None, depth_colormap

    return keypoints_out, depth_colormap


def transformPixelToCameraCoords(coords):  # Pixel to Camera Coordinates
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
    base_vector_x = getVector([keypoints[0][0], keypoints[0][1], keypoints[0][2]],
                              [keypoints[10][0], keypoints[10][1], keypoints[10][2]])
    base_vector_y = getVector([keypoints[0][0], keypoints[0][1], keypoints[0][2]],
                              [keypoints[2][0], keypoints[2][1], keypoints[2][2]])

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
        cv2.circle(npimg, center, 2, common.CocoColors[i], thickness=1, lineType=4, shift=0)
    return npimg, centers


logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
filename_prefix = '../../data/recording_samples/videos/keypoints_movement_'
depth_intrins = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')

    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=432x368, no resize: 0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=1.5,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_v2_large',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')

    parser.add_argument('--videoLocation', type=str,
                        help='location of bag file recorded by RealSense camera')

    parser.add_argument('--tensorrt', type=str, default="True",
                        help='for tensorrt process.')

    parser.add_argument('--movementDetector', type=str, default="../../models/movement_classifier_20190927-180110.h5",
                        help='path to movement-detector model.')

    parser.add_argument('--recordingResolution', type=str, default="640x480",
                        help='Resolution of the recorded video file')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        estimator_obj = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        estimator_obj = TfPoseEstimator(get_graph_path(args.model), target_size=(368, 368), trt_bool=str2bool(args.tensorrt))

    frame_buffer = [[],[],[],[],[],[],[],[],[],[],[]]

    logger.debug('initialization %s : %s' % ('Movement Detector', args.movementDetector))
    movement_detector = tf.keras.models.load_model(args.movementDetector)
    logger.debug(movement_detector.summary())

    print("creating rs stuff")
    config = rs.config()
    rs.config.enable_device_from_file(config, args.videoLocation , repeat_playback=False)
    pipeline = rs.pipeline()
    align_to = rs.stream.color
    align = rs.align(align_to)

    if args.tensorrt == 'True':
        modelsuffix = "_" + args.model + '_tensorRT'
    else:
        modelsuffix = "_" + args.model

    res_width = int(args.recordingResolution.split('x')[0])
    res_height = int(args.recordingResolution.split('x')[1])
    res=(res_width, res_height)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V') #codec
    filename = args.videoLocation.split('.')[0].split('/')[-1]
    location = '../../data/recording_samples/videos/'
    out_filename = location + filename + modelsuffix + '.mp4'
    out = cv2.VideoWriter(out_filename, fourcc, 15, res)

    print("setup complete.")
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    try:
        i = 0
        while True:
            frame = pipeline.wait_for_frames()
            playback.pause()
            logger.info('reading frame: ' + str(i))
            aligned_frames = align.process(frame)

            depth_frame = aligned_frames.get_depth_frame()

            # calibriated internal parameters from RealSense
            depth_intrins = depth_frame.profile.as_video_stream_profile().intrinsics
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.2), cv2.COLORMAP_JET)

            logger.info('image process+')
            humans = estimator_obj.inference(color_image, resize_to_default=(w > 0 and h > 0),
                                             upsample_size=args.resize_out_ratio)

            logger.info('postprocess+')
            if len(humans) > 0:
                human = humans[0]
                color_image, keypoints = draw_human_v2(color_image, human, imgcopy=False)
                keypoint_coords, depth_colormap = getKeyPointCoords(keypoints, depth_frame, depth_colormap)

                if keypoint_coords != None:
                    if len(frame_buffer[10]) == 45:
                        for keypoint in range(11):
                            frame_buffer[keypoint] = np.delete(frame_buffer[keypoint], [0, 1, 2])
                    for keypoint in range(len(keypoint_coords)):
                        frame_buffer[keypoint] = np.append(frame_buffer[keypoint], keypoint_coords[keypoint])

            prediction = [0]
            if len(frame_buffer[10]) == 45:
                data_in = np.asarray([np.expand_dims(frame_buffer, axis=3)])
                prediction = movement_detector.predict(data_in)

            if prediction[0] == 1:
                logger.info('Boxing')
                cv2.putText(color_image, 'Boxing', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(color_image,
                        "%f" % (1.0 / (time.time() - fps_time)),
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(color_image,
                        modelsuffix,
                        (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            # images = np.hstack((color_image,  depth_colormap))

            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break
            logger.info('finished+')
            # out.write(images)
            out.write(color_image)

            playback.resume()
            i += 1
    except Exception as err:
        logger.critical(err)
        pass
    finally:
        pipeline.stop()
        out.release()
        out = None
