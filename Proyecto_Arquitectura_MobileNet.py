import pandas as pd 
import numpy as np 
import os
import keras
import matplotlib.pyplot as plt 
import tensorflow as tf 
from keras import applications
from keras.utils import to_categorical
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model 
from keras.optimizers import Adam
from mlxtend.evaluate import confusion_matrix
from keras import backend as keras 
import intertools

k.clear_session()

base_model = MobileNet(weights = 'imagenet', include_top = False, input_shape = (160,160,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x)
x = Dense(1024, activation = 'relu')(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.5)(x)
preds = Dense(2, activatio = 'softmax')(x)

model = Model(inputs = base_model.input, outputs = preds)

for i , layer in enumerate(model.layers):
	print(i, layer.name)

model.summary()


for layer in model.layers[:20]:
	layer.trainable = False

for layer in model.layers[20:]:
	layer.trainable = True

