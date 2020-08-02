#!/usr/bin/env python
# coding: utf-8

# In[71]:


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
from keras.applications import MobileNet, vgg16
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model 
from keras.optimizers import Adam
from mlxtend.evaluate import confusion_matrix
from keras import backend as K
from itertools import product
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions


# In[72]:


K.clear_session()


# In[73]:


#Especificando las rutas del data set para los archivos de entrenamiento y de validadion de la red neuronal
path_entrenamiento = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_entrenamiento'
path_validacion = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_validacion'
path_covid = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_entrenamiento/covid'
path_nocovid = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_entrenamiento/nocovid'
path_pruebas = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/imagenes_prueba'
path_covid_val = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_validacion/covid'
path_nocovid_val = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/img_validacion/nocovid'


# In[74]:


#HiperParametros
epocas = 40
altura, longitud = 224, 224
batch_size = 20
clases = 2
learning_rate = 0.0005
optimizer = Adam(lr=learning_rate)


# In[75]:


#Funciones para el preprocesamiento de imagenes
data_gen_entrenamiento = ImageDataGenerator(
	rescale = 1./255,
	shear_range = 0.1,
	zoom_range = 0.3,
    rotation_range = 15,
	horizontal_flip = True,
    brightness_range = [0.9,1.1]
)

data_gen_validacion = ImageDataGenerator(
	rescale = 1./255
	)

data_prueba = ImageDataGenerator(rescale = 1./255)


# In[76]:


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


# In[77]:


print(imagen_entrenamiento.class_indices)


# In[78]:


pasos_por_epoca = ((len(os.listdir(path_covid))+len(os.listdir(path_nocovid)))/batch_size)-1
pasos_de_validacion = ((len(os.listdir(path_covid_val))+len(os.listdir(path_nocovid_val)))/batch_size)-1


# In[79]:


#Implementacion de la arquitectura
VGG = vgg16.VGG16(include_top= False, weights= 'imagenet', input_shape=(longitud,altura,3))

#MobileNetV2(weights = 'imagenet', include_top = False, input_shape = (longitud,altura,3))


# In[80]:


VGG.trainable = False
model = keras.Sequential([
    VGG,
    keras.layers.Flatten(),
    keras.layers.Dense(units = 256, activation = 'relu'),
    keras.layers.Dropout(0.40),
    keras.layers.Dense(units = 64, activation = 'relu'),
    keras.layers.Dense(2, activation = 'softmax', kernel_initializer='random_uniform', bias_initializer='zeros'),
])
#x = base_model
#x = GlobalAveragePooling2D()(x)
#x = fc2()(x)
#x = Dense(2048, activation = 'relu')(x)
#x = Dropout(0.50)(x)
#x = BatchNormalization()(x)
#x = Dense(1024, activation = 'relu')(x)
#x = Dropout(0.40)(x)
#x = Dense(512, activation = 'relu')(x)
#x = Dense(256, activation = 'relu')(x)
#x = Dropout(0.10)(x)
#x = Dense(128, activation = 'relu')(x)
#x = Dense(64, activation = 'relu')(x)
#x = Dense(32, activation = 'relu')(x)
#x = Dropout(0.10)(x)
#x = Dense(16, activation = 'relu')(x)
#x = BatchNormalization()(x)
#preds = Dense(2, activation = 'softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
#model = Model(inputs = base_model.input, outputs = preds)


# In[81]:


#print(model.layers[10:])


# In[82]:


for i , layer in enumerate(model.layers):
	print(i, layer.name)


# In[83]:


model.summary()


# In[84]:


for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)


# In[85]:


model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor = 'val_loss',patience = 4, restore_best_weights = True)
H = model.fit(imagen_entrenamiento, steps_per_epoch = pasos_por_epoca , epochs = epocas, validation_data = imagen_validacion, validation_steps = pasos_de_validacion)
#Guardando el modelo
dir = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/VGG16'

if not os.path.exists(dir):
	os.mkdir(dir)

model.save('E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/VGG16/modelo_vgg16.h5')
model.save_weights('E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/VGG16/pesos_modelo_vgg16.h5')
print('Modelo Guardado!')


# In[ ]:





# In[91]:


N = epocas
plt.style.use("ggplot")
plt.figure()
#plt.plot(np.arange(0,N),H.history["loss"], label = "train_loss")
#plt.plot(np.arange(0,N),H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0,N),H.history["accuracy"], label = "train_acc")
plt.plot(np.arange(0,N),H.history["val_accuracy"], label = "val_acc")
plt.title("Precisión de entrenamiento y precisión de validación")
#plt.title("Pérdida de entrenamiento y pérdida de validación")
#plt.title("Entrenamiento Accurancy en el Dataset")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.legend(loc = "lower left")
plt.savefig("plot.png")


# In[93]:


STEP_SIZE_TEST = prueba_generador.n//prueba_generador.batch_size
prueba_generador.reset()
pred = model.predict(prueba_generador, steps= STEP_SIZE_TEST,verbose = 1)


# In[94]:


predicted_class_indices = np.argmax(pred, axis = 1)


# In[95]:


print (predicted_class_indices)
print(len(predicted_class_indices))
print (type(predicted_class_indices))


# In[96]:


labels = (imagen_entrenamiento.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


# In[97]:


filenames = prueba_generador.filenames
results = pd.DataFrame({"Filename":filenames, "Predictions":predictions})
results.to_csv("E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/resultados_VGG16.csv", index = False)


# In[98]:


print(len(filenames))


# In[99]:


real_class_indices=[]
for i in range(0,len(filenames)):
    your_path = filenames[i]
    path_list = your_path.split(os.sep)
    if("SarsCov2" in path_list[1]):
        real_class_indices.append(0)
    if("nocovid" in path_list[1]):
        real_class_indices.append(1)
print (real_class_indices)
print(len(real_class_indices))
real_class_indices = np.array(real_class_indices)
print (type(real_class_indices))


# In[100]:


cm = confusion_matrix(real_class_indices, predicted_class_indices)
from itertools import product


# In[101]:


def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
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
    for i,j in product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
                 horizontalalignment="center",
                 color="white" if cm[i,j]>thresh else "black")
    plt.tight_layout()
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta predecida')


# In[102]:


cm_plot_labels = imagen_entrenamiento.class_indices
plot_confusion_matrix(cm,cm_plot_labels, title = 'Matriz de confusion VGG16')


# In[ ]:




