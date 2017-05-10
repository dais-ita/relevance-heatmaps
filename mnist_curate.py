import numpy as np

import os


image_y = [line.rstrip('\n') for line in open("mnist_train.csv", "r")]

print(image_y[0])


image_data = image_y[0].split(",")


label = image_data.pop(0)

print(label)
print(image_data)

image = np.array(image_data)

image = image.reshape(28,28)

print(image)
