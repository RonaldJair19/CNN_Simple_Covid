#!/usr/bin/env python
# coding: utf-8

# In[79]:


import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model


# In[80]:


from os import system


# In[81]:


longitud, altura = 224, 224
modelo_mobilenet = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/ModeloMobileNetV2/modelo_proyecto_MobileNetV2.h5'
pesos_mobilenet = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/ModeloMobileNetV2/pesos__proyecto_MobileNetV2.h5'
modelo_vgg16 = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/ModeloVGG16/modelo_vgg16.h5'
pesos_vgg16 = 'E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/ModeloVGG16/pesos_modelo_vgg16.h5'
#cnn = load_model(modelo)
#cnn.load_weights(pesos)



# In[82]:


def mobilenet():
    system("cls")
    print("___________________________________________")
    print("Ha seleccionado la arquitectura MobileNetV2")
    print("___________________________________________")
    print("Ingrese el nombre de una imagen contenida en la carpeta: ")
    print("E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/imagenes_prueba/prueba")
    imagen = input("Nombre: ")
    imagen_predecir = "E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/imagenes_prueba/prueba/"+imagen
    #print(imagen_predecir)
    predictor_mobilenet(imagen_predecir)
    


# In[83]:


def vgg16():
    system("cls")
    print("_____________________________________")
    print("Ha seleccionado la arquitectura VGG16")
    print("_____________________________________")
    print("Ingrese el nombre de una imagen contenida en la carpeta: ")
    print("E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/imagenes_prueba/prueba")
    imagen = input("Nombre: ")
    imagen_predecir = "E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/DataSet/imagenes_prueba/prueba/"+imagen
    #print(imagen_predecir)
    predictor_vgg16(imagen_predecir)


# In[84]:


def imprimir():
    print("___________________________________")
    print("Bienvenido al clasificador Covid-19")
    print("___________________________________")
    print()
    print("Porfavor seleccione la red neuronal que desea utilizar para realizar su clasificación:")
    print("1. MobileNetV2")
    print("2. VGG16")
    print("3. <--- Salir")
    print()


# In[85]:


def menu():
    #system("cls")
    salir = False
    opcion = 0
    while not salir:
        imprimir()
        opcion = int(input("Opcion: "))
        if opcion == 1:
            #print ("Opcion 1")
            mobilenet()
        elif opcion == 2:
            vgg16()
        elif opcion == 3:
            salir = True
        else:
            print ("Introduce un numero entre 1 y 3")
    print ("Gracias por utilizar este clasificador!")


# In[86]:


def predictor_mobilenet(file):
    cnn = load_model(modelo_mobilenet)
    cnn.load_weights(pesos_mobilenet)
    x = load_img(file, target_size = (longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis = 0)
    arreglo = cnn.predict(x)
    resultado = arreglo[0]
    respuesta = np.argmax(resultado)
    if respuesta == 0:
        print ('Positivo para Covid!')
    elif respuesta == 1:
        print ('No es afección por Covid!')
    return respuesta


# In[87]:


def predictor_vgg16(file):
    cnn = load_model(modelo_vgg16)
    cnn.load_weights(pesos_vgg16)
    x = load_img(file, target_size = (longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis = 0)
    arreglo = cnn.predict(x)
    resultado = arreglo[0]
    respuesta = np.argmax(resultado)
    if respuesta == 0:
        print ('Positivo para Covid!')
    elif respuesta == 1:
        print ('No es afección por Covid!')
    return respuesta


# In[88]:


if __name__ == '__main__':
    menu()


# In[14]:


#predictor('E:/Documentos/UTP/Cuarto_Anio/Sistemas_basados_en_el_conocimiento/Proyecto/imagenes_internet/ejemplonormal (4).jpg')


# In[ ]:




