#!/usr/bin/env python
# coding: utf-8

# In[38]:


import sys
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation, Convolution2D, MaxPooling2D
from tensorflow.keras import backend as K 
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mlxtend.evaluate import confusion_matrix


# In[28]:


K.clear_session()

#Especificando las rutas del data set para los archivos de entrenamiento y de validadion de la red neuronal
path_entrenamiento = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_entrenamiento'
path_validacion = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_validacion'
path_covid = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_entrenamiento/covid'
path_nocovid = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_entrenamiento/nocovid'


# In[19]:


#print('Total de imagenes de entrenamiento:', len(os.listdir('E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_entrenamiento/nocovid')))


# In[29]:


#Declarando los parametros de la red neuronal
epocas = 20
altura, longitud = 250, 250
batch_size = 20
#pasos = 500
FConv1 = 32
FConv2 = 64
tam_fil1 = (3,3)
tam_fil2 = (2,2)
tam_pool = (2,2)
clases = 2
lr = 0.0005


# In[30]:


#Funciones para el preprocesamiento de imagenes
data_gen_entrenamiento = ImageDataGenerator(
	rescale = 1./255,
	shear_range = 0.3,
	zoom_range = 0.3,
	horizontal_flip = True
)

data_gen_validacion = ImageDataGenerator(
	rescale = 1./225
	)


# In[31]:


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


# In[34]:


#steps_per_epoch = imagen_entrenamiento.n
#validation_steps = imagen_validacion.n
#print (steps_per_epoch)
#print (validation_steps)
pasos_por_epoca = ((len(os.listdir(path_covid))+len(os.listdir(path_nocovid)))/batch_size)-1
pasos_de_validacion = 180/20


# In[ ]:


#Editar las capas de convolucion, hay que agregar mas capas debido al tamaño de las imagenes
#Se crea la red convolucional
cnn = Sequential()
#Primera capa - Capa de partida
cnn.add(Convolution2D(FConv1, tam_fil1, padding = 'same', input_shape = (altura, longitud,3), activation = 'relu'))
#Segunda Capa
cnn.add(MaxPooling2D(pool_size = tam_pool))
#Tercera capa
cnn.add(Convolution2D(FConv2, tam_fil2, padding = 'same', activation = 'relu'))
#Cuarta capa
cnn.add(MaxPooling2D(pool_size = tam_pool))
#Quinta Capa, aplana la imagen
cnn.add(Flatten())
#Sexta capa densa
cnn.add(Dense(256, activation = 'relu'))
#Septima capa, capa de apagado
cnn.add(Dropout(0.5))
#Octava capa densa
cnn.add(Dense(clases, activation = 'softmax'))
#Copilacin
cnn.compile(loss = 'categorical_crossentropy', optimizer = optimizers.Adam(lr = lr, beta_1 = 0.5 ), metrics = ['accuracy'])

H = cnn.fit(imagen_entrenamiento, steps_per_epoch = pasos_por_epoca , epochs = epocas, validation_data = imagen_validacion, validation_steps = pasos_de_validacion)


#Guardando el modelo
dir = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/Modelo2'

if not os.path.exists(dir):
	os.mkdir(dir)

cnn.save('E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/Modelo/modelo_prueba.h5')
cnn.save_weights('E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/Modelo/pesos_modelo.h5')
print('Modelo Guardado!')


# In[40]:


N = epocas
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N),H.history["loss"], label = "train_loss")
plt.plot(np.arange(0,N),H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0,N),H.history["acc"], label = "train_acc")
plt.plot(np.arange(0,N),H.history["val_acc"], label = "val_acc")
plt.title("Entrenamiento Loss and Accurancy en el Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accurancy")
plt.legend(loc = "lower left")
plt.savefig("plot.png")


# In[ ]:




