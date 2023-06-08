import cv2
import numpy as np
import tensorflow as tf


class Classificator:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def classify(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        image = image / 255.0
        prediction = self.model.predict(np.array([image]))

        if np.argmax(prediction):
            return 'Fire'
        return 'No Fire'
