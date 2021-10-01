import numpy as np
import operator
import cv2
import sys, os
from keras.models import load_model
from keras.models import model_from_json
import json

with open('model.json','r') as f:
    model_json = json.load(f)
loaded_model = model_from_json(json.dumps(model_json))
loaded_model.load_weights('epoch5_model.h5')

images = ('A_test.jpg', 'nutu_test.jpeg')

for i in images:
	img = cv2.imread(i)
	img = cv2.resize(img, (200, 200))
	img = np.reshape(img,[1,200,200,3])
	pred = loaded_model.predict(img)
	print(pred)
