import tensorflow as tf
import tensorlayer as tl
import numpy as np
import time

from tensorlayer.models.imagenet_classes import class_names
from openpose_plus.models import get_model
from openpose_plus.inference.common import measure, plot_humans, read_imgfile
from openpose_plus.inference.estimator import TfPoseEstimator

tf.logging.set_verbosity(tf.logging.INFO)
tl.logging.set_verbosity(tl.logging.INFO)


height, width = (368, 432)
vgg = tl.models.vgg19(pretrained=True)
start_time = time.time()

e = measure(lambda: TfPoseEstimator('../../models/vgg19_weights.npz', vgg, target_size=(width, height), data_format=data_format), 'create TfPoseEstimator')
image = measure(lambda: read_imgfile('~/Pictures/human.jpg', width, height, data_format='channels_last'), 'read_imgfile')
humans, heatMap, pafMap = measure(lambda: e.inference(image), 'e.inference')

tl.logging.info('got %d humans from %s' % (len(humans), img_name))
if humans:
	for h in humans:
		tl.logging.debug(h)
if plot:
	if data_format == 'channels_first':
		image = image.transpose([1, 2, 0])
	plot_humans(image, heatMap, pafMap, humans, '%02d' % (idx + 1))

end_time = time.time() - start_time 
print("Endtime: ", end_time)


