# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array
import numpy as np
import time

# tf.keras.backend.set_session(tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0})))

# model = MobileNetV2()
# model = MobileNet()
model = load_model('imagenet_mobilenet.h5')
model.summary()


img = load_img('parrot.jpg', target_size=(224, 224))
arr = img_to_array(img)
arr = arr / 255.
arr_input = np.stack([arr, ])

probs = model.predict(arr_input)
print(np.argmax(probs[0]))

call_num = 100
start = time.time()
for i in range(call_num):
	probs = model.predict(arr_input)
elapsed_time = time.time() - start
print ("time: {0}".format(elapsed_time / call_num * 1000) + "[msec]")

