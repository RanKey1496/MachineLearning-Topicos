# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 18:34:05 2017

@author: RanKey
"""
# In[1]:

#importamos las librerías
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression #Importa el modelo de clasificación
from sklearn.cross_validation import train_test_split #Importa las funciones de validación cruzada
from sklearn.preprocessing import StandardScaler #Importa las funciones de preprocesamineto

from sklearn import datasets
cancer = datasets.load_breast_cancer()

# In[2]:
# Regresión
datos = cancer.data[:,:29]
y = cancer.target

caracteristicas = np.shape(datos)[1]
clasificadas = [[0] * 4 for i in range(caracteristicas)]

def first_letter(line):
    words = line.split()
    letters = [word[0] for word in words]
    return ("".join(letters))

def target(X):
    for i in range(caracteristicas):
        selector = [x for x in range(X.shape[1]) if x != i]
        x = X[:,selector]
        
        #Hacemos la división del conjunto de entrenamiento y el conjunto de validación
        X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
        
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_norm = sc.transform(X_train)
        X_test_norm = sc.transform(X_test)
        
        #Ahora entrenamos el clasificador
        clasificador = LogisticRegression(C=1, random_state=0) #C es el parámetro de regularización
        clasificador.fit(X_train_norm, y_train) #Entrenamiento del clasificador
        
        #Para validar el clasificador
        y_pred = clasificador.predict(X_test_norm)
        correct = (y_pred - y_test) == 0
        log_acc = sum(correct) / y_test.size
        clasificadas[i][0] = (y_test != y_pred).sum()
        clasificadas[i][1] = format(log_acc, '.3f')
        clasificadas[i][2] = list(cancer.feature_names)[i]
        clasificadas[i][3] = first_letter(list(cancer.feature_names)[i])
        
target(datos)
sorted(clasificadas, key=lambda k: k[0])
clasificadas.sort(key=lambda x: x[1], reverse=True) 

people = [row[3] for row in clasificadas]
score = [row[0] for row in clasificadas]
x_pos = np.arange(len(people))

# calculate slope and intercept for the linear trend line
slope, intercept = np.polyfit(x_pos, score, 1)
trendline = intercept + (slope * x_pos)

plt.plot(x_pos, trendline, color='red', linestyle='--')    
plt.bar(x_pos, score,align='center')
plt.xticks(x_pos, people) 
plt.ylabel('Errores')
plt.xlabel('Caracteristicas')
plt.show()