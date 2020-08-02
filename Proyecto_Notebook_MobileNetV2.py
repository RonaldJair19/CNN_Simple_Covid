#!/usr/bin/env python
# coding: utf-8

# In[57]:


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
from keras.layers import BatchNormalization
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model 
from keras.optimizers import Adam
from mlxtend.evaluate import confusion_matrix
from keras import backend as K
from itertools import product
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions


# In[58]:


K.clear_session()


# In[59]:


#Especificando las rutas del data set para los archivos de entrenamiento y de validadion de la red neuronal
path_entrenamiento = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_entrenamiento'
path_validacion = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_validacion'
path_covid = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_entrenamiento/covid'
path_nocovid = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_entrenamiento/nocovid'
path_pruebas = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/imagenes_prueba'
path_covid_val = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_validacion/covid'
path_nocovid_val = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_validacion/nocovid'


# In[60]:


#HiperParametros
epocas = 100
altura, longitud = 224, 224
batch_size = 20
clases = 2
learning_rate = 0.0003
optimizer = Adam(lr=learning_rate)


# In[61]:


#Funciones para el preprocesamiento de imagenes
data_gen_entrenamiento = ImageDataGenerator(
	rescale = 1./255,
	shear_range = 0.1,
	zoom_range = 0.4,
    rotation_range = 30,
	horizontal_flip = True,
    #brightness_range = [0.9,1.1]
)

data_gen_validacion = ImageDataGenerator(
	rescale = 1./255
	)

data_prueba = ImageDataGenerator(rescale = 1./255)


# In[62]:


imagen_entrenamiento = data_gen_entrenamiento.flow_from_directory(
	path_entrenamiento,
	target_size = (altura, longitud),
	batch_size = batch_size,
	class_mode = 'categorical'
	)

imagen_validacion = data_gen_validacion.flow_from_directory(
	path_validacion,
	target_size = (altura, longitud),
	batch_size = batch_size,
	class_mode = 'categorical'
	)

prueba_generador = data_prueba.flow_from_directory(
    path_pruebas,
    target_size = (altura, longitud),
    color_mode = "rgb",
    batch_size = 1,
    class_mode = None,
)


# In[63]:


print(imagen_entrenamiento.class_indices)


# In[64]:


pasos_por_epoca = ((len(os.listdir(path_covid))+len(os.listdir(path_nocovid)))/batch_size)-1
pasos_de_validacion = ((len(os.listdir(path_covid_val))+len(os.listdir(path_nocovid_val)))/batch_size)-1


# In[65]:


#Implementacion de la arquitectura
base_model = MobileNetV2(weights = 'imagenet', include_top = False, input_shape = (longitud,altura,3))


# In[66]:


x = base_model.output
x = GlobalAveragePooling2D()(x)
#x = Dense(2048, activation = 'relu')(x)
#x = Dropout(0.50)(x)
x = BatchNormalization()(x)
#x = Dense(1024, activation = 'relu')(x)
#x = Dropout(0.40)(x)
x = Dense(512, activation = 'relu')(x)
#x = Dense(256, activation = 'relu')(x)
x = Dropout(0.40)(x)
#x = Dense(128, activation = 'relu')(x)
#x = Dense(64, activation = 'relu')(x)
#x = Dense(32, activation = 'relu')(x)
#x = Dropout(0.10)(x)
#x = Dense(16, activation = 'relu')(x)
x = BatchNormalization()(x)
preds = Dense(2, activation = 'softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)

model = Model(inputs = base_model.input, outputs = preds)


# In[45]:


MNet = MobileNetV2(weights = 'imagenet', include_top = False, input_shape = (longitud,altura,3))
MNet.summary()


# In[35]:


#Comprobando capas originales de la MobileNetV2
for i , layer in enumerate(MNet.layers):
	print(i, layer.name)


# In[17]:


for i , layer in enumerate(model.layers):
	print(i, layer.name)


# In[46]:


model.summary()


# In[67]:


for layer in model.layers[:155]:
	layer.trainable = False

for layer in model.layers[155:]:
	layer.trainable = True


# In[68]:


for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)


# In[ ]:





# In[69]:


model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor = 'val_loss',patience = 4, restore_best_weights = True)

H = model.fit(imagen_entrenamiento, steps_per_epoch = pasos_por_epoca , epochs = epocas, validation_data = imagen_validacion, validation_steps = pasos_de_validacion)
#Guardando el modelo
dir = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/ModeloMobileNetV2'

if not os.path.exists(dir):
	os.mkdir(dir)

model.save('E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/ModeloMobileNetV2/modelo_proyecto_MobileNetV2.h5')
model.save_weights('E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/ModeloMobileNetV2/pesos__proyecto_MobileNetV2.h5')
print('Modelo Guardado!')


# In[ ]:





# In[71]:


N = epocas
plt.style.use("ggplot")
plt.figure()
#plt.plot(np.arange(0,N),H.history["loss"], label = "train_loss")
#plt.plot(np.arange(0,N),H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0,N),H.history["accuracy"], label = "train_acc")
plt.plot(np.arange(0,N),H.history["val_accuracy"], label = "val_acc")
plt.title("Precisión de entrenamiento y precisión de validación")
#plt.title("Pérdida de entrenamiento y pérdida de validación")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accurancy")
plt.legend(loc = "lower left")
plt.savefig("plot.png")


# In[72]:


STEP_SIZE_TEST = prueba_generador.n//prueba_generador.batch_size
prueba_generador.reset()
pred = model.predict(prueba_generador, steps= STEP_SIZE_TEST,verbose = 1)


# In[73]:


predicted_class_indices = np.argmax(pred, axis = 1)


# In[74]:


print (predicted_class_indices)
print(len(predicted_class_indices))
print (type(predicted_class_indices))


# In[75]:


labels = (imagen_entrenamiento.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


# In[76]:


filenames = prueba_generador.filenames
results = pd.DataFrame({"Filename":filenames, "Predictions":predictions})
results.to_csv("E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/resultados_MobileNetV2.csv", index = False)


# In[ ]:




