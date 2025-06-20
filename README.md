# Implementación de CNN Forward en C++

Este proyecto implementa el paso forward de una Red Neuronal Convolucional (CNN) en C++ para el conjunto de datos Fashion-MNIST. La implementación incluye capas convolucionales, padding, kernels, stride, función de activación ReLU, capas de pooling (Max, Min, Average) y un Perceptrón Multicapa (MLP).

## Tabla de Contenidos

- [Introducción](#introducción)
- [Capa Convolucional](#capa-convolucional)
- [Padding](#padding)
- [Kernel](#kernel)
- [Convolución](#convolución)
- [Stride](#stride)
- [Función de Activación ReLU](#función-de-activación-relu)
- [Capa de Pooling](#capa-de-pooling)
- [MLP](#mlp)
- [Conexión de Salida de Capas Convolucionales](#conexión-de-salida-de-capas-convolucionales)
- [Resultados](#resultados)
- [Conclusión](#conclusión)

## Introducción

Este proyecto tiene como objetivo demostrar la implementación del paso forward de una CNN en C++. La CNN está diseñada para clasificar imágenes del conjunto de datos Fashion-MNIST, que consiste en imágenes en escala de grises de 28x28 píxeles de 10 categorías diferentes de moda.

## Capa Convolucional

La capa convolucional es fundamental en una CNN, ya que se encarga de extraer características de las imágenes de entrada mediante la aplicación de filtros o kernels. Utiliza operaciones de convolución para transformar la imagen de entrada en mapas de características.

## Padding

El padding se utiliza para agregar bordes adicionales a las imágenes de entrada. Esto permite controlar el tamaño de la salida de las capas convolucionales y mejorar el rendimiento en los bordes de la imagen. Se implementa agregando ceros alrededor de la imagen para mantener las dimensiones espaciales después de la convolución.

## Kernel

Los kernels son matrices de pesos que se aplican a la imagen de entrada para extraer características. Cada kernel se desliza sobre la imagen para realizar la operación de convolución, y se inicializan aleatoriamente para comenzar el proceso de aprendizaje.

## Convolución

La convolución es el proceso de aplicar los kernels a la imagen de entrada para producir mapas de características. Este proceso implica multiplicar y sumar los valores de la imagen y los kernels, lo que permite a la red detectar características como bordes, texturas, etc.

## Stride

El stride define el paso con el que se mueve el kernel sobre la imagen de entrada. Un stride más grande reduce el tamaño de la salida y puede ayudar a reducir la dimensionalidad, lo que a su vez disminuye la cantidad de cálculos necesarios.

## Función de Activación ReLU

La función de activación ReLU (Rectified Linear Unit) se utiliza para introducir no linealidad en el modelo. Esto permite que la red aprenda funciones más complejas y es esencial para el aprendizaje profundo, ya que ayuda a mitigar el problema del gradiente vanishing.

## Capa de Pooling

Las capas de pooling se utilizan para reducir la dimensionalidad de los mapas de características, lo que ayuda a reducir la cantidad de parámetros y cálculos en la red. Se implementan tres tipos de pooling: Max, Min y Average, cada uno con diferentes efectos en la reducción de dimensionalidad.

## MLP

El Perceptrón Multicapa (MLP) es una red neuronal completamente conectada que se utiliza para la clasificación final. Toma los mapas de características extraídos por las capas convolucionales y pooling, y produce la salida final de la red, que en este caso es la clasificación de la imagen de entrada.

## Conexión de Salida de Capas Convolucionales

La conexión de salida de las capas convolucionales a un MLP implica aplanar los mapas de características en un vector unidimensional. Este vector se utiliza como entrada para el MLP, que luego realiza la clasificación final.

## Resultados
![image](https://github.com/user-attachments/assets/832c551e-a399-4227-a6d1-dd798c514636)
![image](https://github.com/user-attachments/assets/1a962cc6-df71-49ab-9dd0-65bf21da10c1)
![image](https://github.com/user-attachments/assets/99a364db-f40f-4aaa-bbef-935594dba14c)
![image](https://github.com/user-attachments/assets/1e168cfc-63e9-43d8-ba92-8a7797306354)



## Conclusión

Este proyecto implementa con éxito el paso forward de una CNN en C++ para el conjunto de datos Fashion-MNIST. La implementación incluye todos los componentes necesarios de una CNN, como capas convolucionales, padding, kernels, stride, función de activación ReLU, capas de pooling y un MLP para la clasificación. El código está estructurado para ser modular y fácil de entender, lo que lo convierte en un buen punto de partida para una mayor exploración y desarrollo.
