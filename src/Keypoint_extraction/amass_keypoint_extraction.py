#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:19:50 2019

@author: jetson-admin
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
	base_vector_x = getVector(keypoints[0], keypoints[12])	
	base_vector_y = getVector(keypoints[0], keypoints[2])
	if np.linalg.norm(base_vector_x) != 0:
		base_vector_x_norm = base_vector_x / np.linalg.norm(base_vector_x)
                
	if np.linalg.norm(base_vector_y) != 0:
		base_vector_y_norm = base_vector_y / np.linalg.norm(base_vector_y)

	base_vector_z_norm = np.cross(base_vector_x_norm, base_vector_y_norm)
    
	return base_vector_x_norm, base_vector_y_norm, base_vector_z_norm

def convertToTfPoseModel(modelKeypoints):
	export_keypoints = []
	for i in relevant_keypoints:
		export_keypoints.append(modelKeypoints[i])
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

sequence_path = "../../amass/HumanEva/S1/Box_1_poses.npz"
model_path = "../../amass/smplh/female/model.npz"
filename_prefix = '../../data/movement/keypoints_movement_'

#order is important to match keypoint order of tf-pose model
relevant_keypoints = [15,17,19,21,16,18,20,2,5,8,1,4,7]

sequence = np.load(sequence_path)
modelnpz = np.load(model_path)
modelKeys = list(modelnpz.keys())
modelValues = list(modelnpz.values())

model_dict = {}
for i in range(len(modelKeys)):
	model_dict[modelKeys[i]] = modelValues[i]


model_dict['bs_type'] = 'lrotmin'
model_dict['bs_style'] =  'lbs'
model_dict['betas'] = sequence['betas']
boxing_movement = []


for pose in range(int(len(sequence['poses'])/17)):
	model_dict['pose'] = sequence['poses'][pose*17]
	model = load_model(model_dict)
	boxing_movement.append(model)
	export_keypoints = convertToTfPoseModel(model.J_transformed)
	filename = filename_prefix + str(pose) + '.xyz'
	print('writing file ' + filename)
	f= open(filename,"w+")
	for keypoint in export_keypoints:
		x,y,z = keypoint[0],keypoint[1],keypoint[2]
		text = ' '+str(x)+' '+str(y)+' '+str(z)
		f.write(text+'\n')
	f.close()

