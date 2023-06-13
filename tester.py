#import sys; args = sys.argv[1:]
import time
from collections import deque
from heapq import heappush, heappop, heapify
import numpy as np
import math
import random
import tensorflow as tf
from tensorflow import keras
fd = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = fd.load_data()
x_train, x_test = x_train/255, x_test/255
print(y_train.shape)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'softmax')
])
print(random.choice(('U', 'L', 'D', 'R')))
# print('x')
# model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# print('x')
# model.fit(x_train, y_train, epochs = 5)
# print('x')