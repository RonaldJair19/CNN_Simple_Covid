#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model


# In[13]:


longitud, altura = 224, 224
modelo = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/ModeloMobileNet/modelo_prueba.h5'
pesos = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/ModeloMobileNet/pesos_modelo.h5'
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


# In[14]:


predictor('E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/imagenes_internet/ejemplonormal (4).jpg')


# In[ ]:




