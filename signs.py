#Signs by Sameer and Samarth
#Classification Code

#from keras.models import load_model
#hand_sign = load_model('asl-alphabet/signs.h5')

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout

#Intialize
hand_sign = Sequential()

#First conv layer
hand_sign.add(Convolution2D(32, (3, 3), input_shape = (200,200, 3), activation = "relu"))
hand_sign.add(MaxPooling2D(pool_size = (2, 2)))

#Second conv layer
hand_sign.add(Convolution2D(32, (3,3), activation = "relu"))
hand_sign.add(MaxPooling2D(pool_size = (2,2)))

#Third conv layer
hand_sign.add(Convolution2D(32, (3,3), activation = "relu"))
hand_sign.add(MaxPooling2D(pool_size = (2,2)))

hand_sign.add(Convolution2D(32, (3,3), activation = "relu"))
hand_sign.add(MaxPooling2D(pool_size = (2,2)))

#Flattening
hand_sign.add(Flatten())
hand_sign.add(Dropout(0.1))
hand_sign.add(Dense(activation = "softmax", units = 29))

#Compiling
hand_sign.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

#Image Preprocessing, Code from Keras Documentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'asl_alphabet_train',
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'asl_alphabet_test',
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical')

hand_sign.fit_generator(
        training_set,
        steps_per_epoch=2400/32,
        epochs=75,
        validation_data=test_set,
        validation_steps=600/32)

hand_sign.save('signs.h5')
print("Model is saving...")

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('asl_alphabet_test/J/J36.jpg',target_size=(200, 200))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = hand_sign.predict(test_image)
training_set.class_indices
