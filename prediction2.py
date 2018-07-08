
from keras.models import model_from_json
from keras.preprocessing import image
#from cnn.py import training_set
import matplotlib.pyplot as plt
from matplotlib import ticker
import cv2
import numpy as np


json_file = open('CatOrDog.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("CorDweights.h5")
print("Loaded model from disk")

'''loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
'''
loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

img_pred = image.load_img('/home/mr-robot/Documents/neuronal_networks/perceptron/Cats-vs-Dogs-Classification-CNN-Keras-/dataset/single_prediction/cat_or_dog_1.jpg', target_size = (150, 150))
plt.figure(figsize=(5,5))
plt.imshow(img_pred)
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)
rslt = loaded_model.predict(img_pred)

for i in range(0,1):
    if rslt[i,0] == 1:
        print('I am {:.2%} sure this is a Dog'.format(rslt[i][0]))
    else:
        print('I am {:.2%} sure this is a Cat'.format(1-rslt[i][0]))

#ind = training_set.class_indices
#
# if rslt[0][0] == 1:
#     prediction = "dog"
#     #print("Creo que es un perro")
#     print('I am sure this is a Dog')
# else:
#     prediction = "cat"
#     print('I am sure this is a Cat')


plt.show()
