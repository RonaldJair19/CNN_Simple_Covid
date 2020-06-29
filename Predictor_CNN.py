#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model


# In[3]:


longitud, altura = 250, 250
modelo = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/Modelo/modelo_prueba.h5'
pesos = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/Modelo/pesos_modelo.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos)


def predictor(file):
    x = load_img(file, target_size = (longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis = 0)
    arreglo = cnn.predict(x)
    resultado = arreglo[0]
    respuesta = np.argmax(resultado)
    if respuesta == 0:
        print ('Covid!')
    elif respuesta == 1:
        print ('NoCovid!')
    return respuesta


# In[24]:


predictor('E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/Archivos de los dataset/17810_23812_bundle_archive/chest_xray/train/PNEUMONIA/person319_bacteria_1479.jpeg')


# In[ ]:




