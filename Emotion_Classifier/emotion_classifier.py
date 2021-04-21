# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 19:27:59 2021

@author: derph
"""

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(len(train_labels))

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
train_images = train_images.reshape(60000, 28, 28, 1)

test_images = test_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = tf.keras.models.Sequential([
    # Adding a few layers of convolution and Max Pooling
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1), padding="same"),
    tf.keras.layers.MaxPooling2D(2, 2),
    #tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1), padding="same"),
    #tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the filtered input
    tf.keras.layers.Flatten(),
    # Add hidden layers with Dropout or BatchNorm()
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(512, activation='relu'),
    #BatchNorm(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(512, activation='relu'),
    #BatchNorm(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(512, activation='relu'),
    #BatchNorm(),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)