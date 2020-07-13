#!/usr/bin/env python
# coding: utf-8

# In[18]:


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
from keras.applications.mobilenet import preprocess_input


# In[49]:


K.clear_session()

#Especificando las rutas del data set para los archivos de entrenamiento y de validadion de la red neuronal
path_entrenamiento = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_entrenamiento'
path_validacion = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_validacion'
path_covid = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_entrenamiento/covid'
path_nocovid = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_entrenamiento/nocovid'
path_pruebas = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/imagenes_prueba'


# In[50]:


#print('Total de imagenes de entrenamiento:', len(os.listdir('E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_entrenamiento/nocovid')))


# In[51]:


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
clases = 4
lr = 0.0005


# In[52]:


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

data_prueba = ImageDataGenerator(rescale = 1./225)


# In[53]:


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


# In[7]:


print(imagen_entrenamiento.class_indices)


# In[8]:


#steps_per_epoch = imagen_entrenamiento.n
#validation_steps = imagen_validacion.n
#print (steps_per_epoch)
#print (validation_steps)
pasos_por_epoca = ((len(os.listdir(path_covid))+len(os.listdir(path_nocovid)))/batch_size)-1
pasos_de_validacion = 180/20


# In[9]:


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
#Copilacion
cnn.compile(loss = 'categorical_crossentropy', optimizer = optimizers.Adam(lr = lr, beta_1 = 0.5 ), metrics = ['accuracy'])

H = cnn.fit(imagen_entrenamiento, steps_per_epoch = pasos_por_epoca , epochs = epocas, validation_data = imagen_validacion, validation_steps = pasos_de_validacion)


#Guardando el modelo
dir = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/Modelo2'

if not os.path.exists(dir):
	os.mkdir(dir)

cnn.save('E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/Modelo/modelo_prueba.h5')
cnn.save_weights('E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/Modelo/pesos_modelo.h5')
print('Modelo Guardado!')


# In[13]:


N = epocas
plt.style.use("ggplot")
plt.figure()
#plt.plot(np.arange(0,N),H.history["loss"], label = "train_loss")
#plt.plot(np.arange(0,N),H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0,N),H.history["accuracy"], label = "train_acc")
plt.plot(np.arange(0,N),H.history["val_accuracy"], label = "val_acc")
plt.title("Entrenamiento Loss and Accurancy en el Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accurancy")
plt.legend(loc = "lower left")
plt.savefig("plot.png")


# In[39]:


data_prueba = ImageDataGenerator(rescale = 1./225)


# In[40]:


prueba_generador = data_prueba.flow_from_directory(
    directory = "E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/test",
    target_size = (altura, longitud),
    color_mode = "rgb",
    batch_size = 1,
    class_mode = None,
)


# In[54]:


STEP_SIZE_TEST = prueba_generador.n//prueba_generador.batch_size
prueba_generador.reset()
pred = cnn.predict_generator(prueba_generador, steps= STEP_SIZE_TEST,verbose = 1)


# In[55]:


predicted_class_indices = np.argmax(pred, axis = 1)


# In[56]:


print (predicted_class_indices)
print (type(predicted_class_indices))


# In[58]:


labels = (imagen_entrenamiento.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


# In[2]:


filenames = prueba_generador.filenames
results = pd.DataFrame({"Filename":filenames, "Predictions":predictions})
results.to_csv("E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/resultados.csv", index = False)


# In[1]:


real_class_indices = []
for i in range(0, len(filenames)):
    your_path = filenames[i]
    path_list = your_path.split(os.sep)
    if("covid" in path_list[0]):
        real_class_indices.append(0)
    if("ct_covid" in path_list[1]):
        real_class_indices.append(1)
    if("ct_nocovid" in path_list[2]):
        real_class_indices.append(2)
    if("nocovid" in path_list[3]):
        real_class_indices.append(3)
    print (real_class_indices)
    print(type(real_class_indices))
    real_class_indices = np.array(real_class_indices)
    print (type(real_class_indices))


# In[ ]:


cm = confusion_matrix(real_class_indices, predicted_class_indices)


# In[ ]:


def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    plt.inshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print ('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max()/2.
    for i,j in itertools, product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
                 horizontalaligment = "center"
                 color="white" if cm[i,j]>thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        


# In[ ]:


cm_plot_labels = imagen_entrenamiento.class_indices
plot_confusion_matrix(cm,cm_plot_labels, title= 'Matriz de confusion')

