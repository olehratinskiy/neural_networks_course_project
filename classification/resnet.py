import os
import cv2
import numpy as np
import tensorflow as tf
from keras.applications import ResNet50
from keras import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam


class ResNet:
    def __init__(self, types_num, X_train, y_train, X_test, y_test):
        self.types_num = types_num
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.types_num, activation='softmax')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self):
        self.model.fit(self.X_train, self.y_train, batch_size=32, epochs=40, validation_data=(self.X_test, self.y_test))

    def save(self, path):
        self.model.save(f'{path}/fire_classification_model.h5')

    def get_accuracy_metrics(self):
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")


def data_preprocessing(fire_images_path, non_fire_images_path):
    types_num = 2

    fire_data = []
    files = os.listdir(fire_images_path)
    for filename in files:
        path = os.path.join(fire_images_path, filename)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            fire_data.append(img)

    non_fire_data = []
    files = os.listdir(non_fire_images_path)
    for filename in files:
        path = os.path.join(non_fire_images_path, filename)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            non_fire_data.append(img)

    fire_labels = [1 for _ in range(len(fire_data))]
    non_fire_labels = [0 for _ in range(len(non_fire_data))]

    data = np.array(fire_data + non_fire_data)
    labels = np.array(fire_labels + non_fire_labels)

    idxes = np.random.permutation(len(data))
    data = data[idxes]
    labels = labels[idxes]

    size = int(len(data) * 0.8)
    X_train, X_test = data[:size], data[size:]
    y_train, y_test = labels[:size], labels[size:]

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, types_num)
    y_test = tf.keras.utils.to_categorical(y_test, types_num)

    return types_num, X_train, y_train, X_test, y_test
