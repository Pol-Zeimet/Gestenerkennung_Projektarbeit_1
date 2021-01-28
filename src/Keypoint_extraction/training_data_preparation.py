#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:23:05 2019

@author: Pol Zeimet
"""

import numpy as np
from webuser.serialization import load_model

def transformToBodyCoords(rotation_matrix, base_keypoint, coord):
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
	base_vector_x = getVector(keypoints[0], keypoints[10])	
	base_vector_y = getVector(keypoints[0], keypoints[2])
	if np.linalg.norm(base_vector_x) != 0:
		base_vector_x_norm = base_vector_x / np.linalg.norm(base_vector_x)
                
	if np.linalg.norm(base_vector_y) != 0:
		base_vector_y_norm = base_vector_y / np.linalg.norm(base_vector_y)

	base_vector_z_norm = np.cross(base_vector_x_norm, base_vector_y_norm)
    
	return base_vector_x_norm, base_vector_y_norm, base_vector_z_norm

def convertToTfPoseModel(model):
    export_keypoints = []
    #order is important to match keypoint order of tf-pose model
    relevant_keypoints = [15,17,19,21,16,18,20,2,1]
    for i in relevant_keypoints:
        export_keypoints.append(model.J_transformed[i])
    neck_x = (float(model.J_transformed[17][0]) + float(model.J_transformed[16][0]))/2
    neck_y = (float(model.J_transformed[17][1]) + float(model.J_transformed[16][1]))/2
    neck_z = (float(model.J_transformed[17][2]) + float(model.J_transformed[16][2]))/2
    base_x = (float(model.J_transformed[2][0]) + float(model.J_transformed[1][0]))/2
    base_y = (float(model.J_transformed[2][1]) + float(model.J_transformed[1][1]))/2
    base_z = (float(model.J_transformed[2][2]) + float(model.J_transformed[1][2]))/2
    base_keypoint = [base_x, base_y, base_z]
    export_keypoints.insert(0, base_keypoint)
    export_keypoints.insert(2, [neck_x, neck_y, neck_z])
    rotation_matrix = np.column_stack(getBodyBaseVectors(export_keypoints))
    for keypoint in range(len(export_keypoints)):
        export_keypoints[keypoint] = transformToBodyCoords(rotation_matrix, base_keypoint, export_keypoints[keypoint])
    return export_keypoints

def generate_Training_file(index, sequence_path, isBoxing):
    print('processing file:' + sequence_path[24:])
    sequence = np.load(sequence_path)
    gender = str(sequence['gender'])
    modelnpz =  np.load(model_paths[gender])
    modelKeys = list(modelnpz.keys())
    modelValues = list(modelnpz.values())

    filename = sequence_path[24:-4].split('_')[0] + '_' + str(index)

    model_dict = {}
    for i in range(len(modelKeys)):
        model_dict[modelKeys[i]] = modelValues[i]
    model_dict['bs_type'] = 'lrotmin'
    model_dict['bs_style'] =  'lbs'
    model_dict['betas'] = sequence['betas']
    boxing_movement = []
    
    if isBoxing:
        cl = 1
    else:
        cl = 0
    data = []
    sample = [[],cl]
    sliding_sample = [[],cl]
    frames_per_sample = 15
    #data is recorded with 120 fps, we only have 7-8, so we reduce the sampling to a 17th 
    
    posecount = int(len(sequence['poses'])/17)
    for pose in range(posecount):    
        model_dict['pose'] = sequence['poses'][pose*17]
        model = load_model(model_dict)
        boxing_movement.append(model)
        export_keypoints = convertToTfPoseModel(model)
        print('\t Processing pose ', str(pose), ' out of ', str(posecount))
        for keypoint in range(len(export_keypoints)):
            if len(sample[0]) != 11:
                sample[0].append(export_keypoints[keypoint].tolist())
            else:
                sample[0][keypoint].extend(export_keypoints[keypoint])
                if len(sample[0][10]) == frames_per_sample*3:
                    data.append(sample)
                    sample = [[],cl]
        
            if pose > int(frames_per_sample/2):
                if len(sliding_sample[0]) != 11:
                    sliding_sample[0].append(export_keypoints[keypoint].tolist())
                else:
                    sliding_sample[0][keypoint].extend(export_keypoints[keypoint])    
                    if len(sliding_sample[0][10]) == frames_per_sample*3:
                        data.append(sliding_sample)
                        sliding_sample = [[],cl]
    data_np = np.asarray(data)
    np.save(filename_location_out + filename, data_np, allow_pickle = True)
    return data

boxing_sequence_paths = ["../../amass/HumanEva/S1/Box_1_poses.npz",
                         "../../amass/HumanEva/S1/Box_3_poses.npz",
                         "../../amass/HumanEva/S2/Box_1_poses.npz",
                         "../../amass/HumanEva/S2/Box_3_poses.npz",
                         "../../amass/HumanEva/S3/Box_1_poses.npz",
                         "../../amass/HumanEva/S3/Box_3_poses.npz"]
not_boxing_sequence_paths = [
        "../../amass/HumanEva/S1/Gestures_1_poses.npz",
        "../../amass/HumanEva/S1/Gestures_3_poses.npz",
        "../../amass/HumanEva/S1/Jog_1_poses.npz",
        "../../amass/HumanEva/S1/Jog_3_poses.npz",
        "../../amass/HumanEva/S1/Static_poses.npz",
        "../../amass/HumanEva/S1/ThrowCatch_1_poses.npz",
        "../../amass/HumanEva/S1/Walking_3_poses.npz",
        "../../amass/HumanEva/S2/Walking_3_poses.npz",
        "../../amass/HumanEva/S2/Walking_1_poses.npz",
        "../../amass/HumanEva/S2/Gestures_1_poses.npz",
        "../../amass/HumanEva/S2/Gestures_3_poses.npz",
        "../../amass/HumanEva/S2/Jog_1_poses.npz",
        "../../amass/HumanEva/S2/Jog_3_poses.npz",
        "../../amass/HumanEva/S2/Static_poses.npz",
        "../../amass/HumanEva/S3/Gestures_1_poses.npz",
        "../../amass/HumanEva/S3/Gestures_3_poses.npz",
        "../../amass/HumanEva/S3/Jog_1_poses.npz",
        "../../amass/HumanEva/S3/Jog_3_poses.npz",
        "../../amass/HumanEva/S3/Static_poses.npz",
        "../../amass/HumanEva/S3/ThrowCatch_1_poses.npz",
        "../../amass/HumanEva/S3/ThrowCatch_3_poses.npz",
        "../../amass/HumanEva/S3/Walking_3_poses.npz",
        ]
model_paths = { 'male' : "../../amass/smplh/male/model.npz", 'female' : "../../amass/smplh/female/model.npz", 'neutral' : "../../amass/smplh/neutral/model.npz"}
filename_location_out = '../../data/movement/training_data/'

total_sample_count = 0

print('non-boxing Data: ')
for index, sequence_path in enumerate(not_boxing_sequence_paths):
    data = generate_Training_file(index, sequence_path, False)
    sample_count = len(data)
    total_sample_count += sample_count
    print('\t extracted samples: ' + str(sample_count)),
    print('\t total non-boxing sample count so far: ' + str(total_sample_count)),
    print('___________________________________________________________________________')

total_boxing_sample_count = 0

print('boxing Data: ')    
for index, sequence_path in enumerate(boxing_sequence_paths):
    data = generate_Training_file(index, sequence_path, True)
    sample_count = len(data)
    total_boxing_sample_count += sample_count
    print('\t extracted samples: ' + str(sample_count)),
    print('\t total boxing sample count so far: ' + str(total_boxing_sample_count)),
    print('___________________________________________________________________________')

print('total samples extracted: ' + str(total_boxing_sample_count + total_sample_count ))
print('\t boxing: ' + str(total_boxing_sample_count ))
print('\t non-boxing: ' + str(total_sample_count ))
