#!/usr/bin/env python
# -*- coding: utf-8 -*-
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import model_from_json


#Dimension de la imagen
img_width, img_height = 150, 150
#Carpeta que almacena las imagenes
#con estas se entrenara la red
train_data_dir = 'data/train'
#carpeta con las muestras de validacion
validation_data_dir = 'data/validation'
#numero de imagenes que se concideran para la validacion
train_samples = 2000
#numero de images que se cocideran para la validacion
validation_samples = 800

#numero de veces que se ejecutara las red sobre el conjunto de entrenamiento
#antes de empezar con la validacion

epoch = 50
#***** Inicio del modelo *****
model= Sequential()
#model.add(Convolution2D(nb_filter, nb_row, nb_col, ))
# nb_filter: Number of convolution filters to use.
# nb_row: Number of rows in the convolution kernel.
# nb_col: Number of columns in the convolution kernel.
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))



# ** FIn del modelo **
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# esta es la mejora de la configuración que utilizaremos para el entrenamiento
# en el que generamos un gran número de imágenes transformadas de manera que el
# modelo puede tratar con una gran variedad de escenarios del mundo real
train_datagen = ImageDataGenerator(
        rescale =1./255,
        shear_range =0.2,
        zoom_range =0.2,
        horizontal_flip =True)


# esta es la mejora de la configuración que utilizaremospara la prueba:
# sólo para reajuste
test_datagen = ImageDataGenerator(rescale=1./255)
# esta sección toma imágenes de la carpeta
# y las pasa al ImageGenerator que crea entonces
# un gran número de versiones transformadas
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

validation_generator =  test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

# aquí es donde se produce el proceso real
# y llevará algún tiempo ejecutar este paso.
model.fit_generator(
        train_generator,
        samples_per_epoch=train_samples,
        nb_epoch=epoch,
        validation_data=validation_generator,
        nb_val_samples=validation_samples)



# for e in range(40):
#     score = model.evaluate(validation_generator, verbose=0)
#     print ('Test loss:', score[0])
#     print ('Test accuracy:', score[1])
clssf = model.to_json()
with open("CatOrDog.json", "w") as json_file:
    json_file.write(clssf)
model.save_weights('CorDweights.h5')
print("model saved to disk....")
