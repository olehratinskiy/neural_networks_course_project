import os
import cv2
import numpy as np
import tensorflow as tf
from keras.applications import ResNet50
from keras import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

types_num = 2
fire_images_path = 'C:\\Users\\23485\\Downloads\\archive (5)\\fire_dataset\\fire_images'
non_fire_images_path = 'C:\\Users\\23485\\Downloads\\archive (5)\\fire_dataset\\non_fire_images'

fire_data = []
files = os.listdir(fire_images_path)
for filename in files:
    path = os.path.join(fire_images_path, filename)
    img = cv2.imread(path)
    if img is not None:
        img = cv2.resize(img, (512, 512))
        fire_data.append(img)

non_fire_data = []
files = os.listdir(non_fire_images_path)
for filename in files:
    path = os.path.join(non_fire_images_path, filename)
    img = cv2.imread(path)
    if img is not None:
        img = cv2.resize(img, (512, 512))
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


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(types_num, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

model.save('fire_classification_model.h5')

# Make predictions on new images
# def predict_image(image_path):
#     img = image.load_img(image_path, target_size=(img_width, img_height))
#     x = image.img_to_array(img)
#     x = tf.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     predictions = model.predict(x)
#     class_index = tf.argmax(predictions, axis=1).numpy()[0]
#     class_labels = ['non-fire', 'fire']
#     result = class_labels[class_index]
#     return result
#
#
# # Example usage
# image_path = 'D:\\Oleg\\neural_networks\\course_work\\static\\image.png'
# prediction = predict_image(image_path)
# print(f"The image is classified as: {prediction}")
