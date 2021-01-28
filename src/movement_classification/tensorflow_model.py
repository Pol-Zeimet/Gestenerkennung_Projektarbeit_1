#!/Documents/projektarbeit/projekt_env Python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:47:42 2019

@author: Pol Zeimet
"""

import tensorflow as tf
import glob
import numpy as np
import time




def load_and_prepare_data(file_location, test_size):
    data = []
    boxing_data = []
    for np_name in glob.glob(file_location + '*.np[yz]'):
        file = np.load(np_name, allow_pickle = True)
        for sequence in file:
            if sequence[1] == 1:
                boxing_data.append([sequence[0], sequence[1]])
            else: 
                data.append([sequence[0], sequence[1]])      
                        
    np.random.shuffle(data)
    np.random.shuffle(boxing_data)
    box_train, box_test = boxing_data[int(test_size * len(boxing_data)):], boxing_data[:int(test_size * len(boxing_data))]
    data_train, data_test = data[int(test_size * len(data)):], data[:int(test_size * len(data))]
    
    train_data = np.append(box_train, data_train, axis = 0)
    test_data = np.append(box_test, data_test, axis = 0)
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    
    x_train, x_test, y_train, y_test = [], [], [], []
    
    for sequence in train_data:
        x_train.append(sequence[0])
        y_train.append(sequence[1])
    
    for sequence in test_data:
        x_test.append(sequence[0])
        y_test.append(sequence[1])

    x_test = np.expand_dims(x_test, axis = 3)
    x_train = np.expand_dims(x_train, axis = 3)
    
    return x_train, x_test, y_train, y_test



def define_and_compile_model():

    model = tf.keras.Sequential() 
    input_conv_layer = tf.keras.layers.Conv2D(
        input_shape = (11,45,1),
        data_format = 'channels_last',
        kernel_size = (11, 9),
        strides = 3,
        padding = 'valid',
        activation = tf.keras.activations.relu,
        kernel_initializer = tf.keras.initializers.glorot_normal(),
        filters = 32
        )
    
    pooling_layer_1 = tf.keras.layers.MaxPool2D(
            data_format = 'channels_last',
            pool_size = (1,3),
            strides = (1,3),
            padding = 'valid'
            )
    conv_layer_2 = tf.keras.layers.Conv2D(
            data_format = 'channels_last',
            kernel_size = (1, 3),
            strides = 1,
            padding = 'valid',
            activation = tf.keras.activations.relu,
            kernel_initializer = tf.keras.initializers.glorot_normal(),
            filters = 64
            )
    
    conv2_flat = tf.keras.layers.Flatten()
    
    dense_layer_1 = tf.keras.layers.Dense(
            activation = tf.keras.activations.relu,
            bias_initializer = tf.keras.initializers.glorot_normal(),
            kernel_initializer = tf.keras.initializers.glorot_normal(),
            units = 100
            )
    
    dense_layer_2 = tf.keras.layers.Dense(
            activation = tf.keras.activations.relu,
            bias_initializer = tf.keras.initializers.glorot_normal(),
            kernel_initializer = tf.keras.initializers.glorot_normal(),
            units = 50
            )
    
    dense_layer_3 = tf.keras.layers.Dense(
            activation = tf.keras.activations.relu,
            bias_initializer = tf.keras.initializers.glorot_normal(),
            kernel_initializer = tf.keras.initializers.glorot_normal(),
            units = 20
            )
    
    dense_layer_4 = tf.keras.layers.Dense(
            activation = tf.keras.activations.relu,
            bias_initializer = tf.keras.initializers.glorot_normal(),
            kernel_initializer = tf.keras.initializers.glorot_normal(),
            units = 20
            )
    
    dense_layer_output = tf.keras.layers.Dense(
            units = 1,
            bias_initializer = tf.keras.initializers.glorot_normal(),
            kernel_initializer = tf.keras.initializers.glorot_normal(),
            activation = 'sigmoid'
            )
    
    model.add(input_conv_layer)
    model.add(pooling_layer_1)
    model.add(conv_layer_2)
    model.add(conv2_flat)
    model.add(dense_layer_1)
    model.add(dense_layer_2)
    model.add(dense_layer_3)
    model.add(dense_layer_4)
    model.add(dense_layer_output)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01), loss='mse', metrics=['accuracy'])
    return model


def accuracy(true, pred):
    correct= 0
    total = len(true)
    for pair in zip(true, pred):
        if pair[0] == pair[1]:
            correct+=1
    return correct/total

def conf_mat(true, pred):
    cl1_correct = 0
    cl1_incorrect = 0
    cl2_correct = 0
    cl2_incorrect = 0
    for pair in zip(true, pred):
        if pair[0] == pair[1]:
            if pair[0] == 0:
                cl1_correct += 1        
            else:
                cl2_correct += 1
        else:
            if pair[0] == 0:
                cl1_incorrect += 1        
            else:
                cl2_incorrect += 1
    return [[cl1_correct, cl1_incorrect],[cl2_incorrect, cl2_correct]]





timestr = time.strftime("%Y%m%d-%H%M%S")
filename = '../../models/movement_classifier_'+timestr

x_train, x_test, y_train, y_test = load_and_prepare_data('../../data/movement/training_data/', 0.15)

model = define_and_compile_model()
model.fit(x_train,y_train, batch_size = 10, epochs = 50, steps_per_epoch = 40, verbose = 1)

print('Evaluating with training Dataset')
pred_train = model.predict(x_train, batch_size = 10,verbose = 1)
print('train accuracy: ' + str(accuracy(y_train, pred_train)))
print('confusion matrix:')
print(conf_mat(y_train, pred_train))

print('__________________________________________________________________________________')

print('Evaluating with test Dataset')
pred_test =  model.predict(x_test, batch_size = 10, verbose = 1)
print('test accuracy: ' + str(accuracy(y_test, pred_test)))
print('confusion matrix:')
print(conf_mat(y_test, pred_test))

model.save(filename+'.h5')
f= open(filename+'.txt',"w+")
text = 'train accuracy: ' + str(accuracy(y_train, pred_train)) + '\n' +'confusion matrix:' + str(conf_mat(y_train, pred_train)) + '\n' +'test accuracy: ' + str(accuracy(y_test, pred_test)) + '\n' +'confusion matrix: ' + str(conf_mat(y_test, pred_test))
f.write(text+'\n')
f.close()

