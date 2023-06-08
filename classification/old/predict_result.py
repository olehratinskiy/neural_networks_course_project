import os
import cv2
import numpy as np
import tensorflow as tf
from keras.applications import ResNet50
from keras import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

model = tf.keras.models.load_model('fire_classification_model.h5')

image_path = "C:\\Users\\23485\\Downloads\\archive (5)\\fire_dataset\\non_fire_images\\non_fire.233.png"
image = cv2.imread(image_path)
image = cv2.resize(image, (512, 512))
image = image / 255.0

prediction = model.predict(np.array([image]))
print(prediction)

class_names = ['Non-Fire', 'Fire']
predicted_class = np.argmax(prediction)
predicted_label = class_names[predicted_class]
print(f"Predicted Label: {predicted_label}")


