import tensorflow as tf
import tensorlayer as tl
import numpy as np
import time
from tensorlayer.models.imagenet_classes import class_names
#using tensorlayer and tensorflow 2.0
vgg = tl.models.vgg16(pretrained=True)
img = tl.vis.read_image('~/Pictures/tiger.jpeg')
img = tf.image.resize(img, [224, 224]) / 255.0

start_time = time.time()
output = vgg(img, is_train=False)
probs = tf.nn.softmax(output)[0].numpy()
print("Endtime: ", start_time - time.time())
preds = (np.argsort(probs)[::-1])[0:5]
for pred in preds:
	print(class_names[pred],probs[pred])

