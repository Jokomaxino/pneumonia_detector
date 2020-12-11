#!/usr/bin/env python3

import tensorflow as tf
import os
import numpy as np
from PIL import Image

print("Loading model...")
doctor = tf.keras.models.load_model('final.h5')
print("Model ready.\n")

path = str(input("input image path: "))

#load image, convert to grayscale, resize to 200x200, convert to array
image = np.asarray(Image.open(path).convert('L').resize((200,200)))
image = image.reshape(1,200,200, 1)

#predict
result = doctor.predict(image)[0,0]

if result >= .5:    print("Pneumonia detected!")
else:               print("The lungs are healthy.")