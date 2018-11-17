import os
import urllib2
import numpy as np
from PIL import Image
from cv2 import resize
from keras.models import Sequential
from vgg16_places_365 import VGG16_Places365
from keras.layers import Dense, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt
import keras.callbacks



train_data_dir = "/home/chris/Desktop/CW_data/training/training"
val_data_dir = "/home/chris/Desktop/CW_data/validation"
batch_size = 10
img_width = 224
img_height = 224
num_classes = 15
num_train_samples = num_classes * 80 
num_val_samples = num_classes * 20 
num_epochs = 10000
augment_data_dir = "/home/chris/Desktop/CW_data/augmented_data"

'''
keras.preprocessing.image.ImageDataGenerator(
 brightness_range=None, 
'''
########### data processing #############
augment_data_gen = ImageDataGenerator(
	data_format='channels_last',
	brightness_range=[0.5,1.5],
	width_shift_range=0.2, 
    height_shift_range=0.2, 
    rescale=1. / 255,
    fill_mode='nearest',
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
simple_data_gen = ImageDataGenerator(rescale=1. / 255)

#default conversion to rgb
train_generator = augment_data_gen.flow_from_directory(
        train_data_dir,
        shuffle=True,
        save_to_dir=augment_data_dir, 
        target_size=(img_width, img_height), 
        batch_size=batch_size,
        class_mode='categorical')

val_generator = simple_data_gen.flow_from_directory(
    val_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


############### load model ########
vgg16 = VGG16_Places365(weights='places')
model = Sequential()
for layer in vgg16.layers[:-1]:
    model.add(layer)

#freeze all the layers before the last (5th) convolutional block
for layer in model.layers:
    if "block5" in layer.name:
        break
    else:
        layer.trainable = False;

model.add(Dense(num_classes, activation='softmax', name="predictions"))

model.summary()

############# callbacks ############
tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', 
	histogram_freq=0, batch_size=batch_size, 
	write_graph=True, write_images=True)

early_stop = keras.callbacks.EarlyStopping(
	monitor='val_loss',	min_delta=0,
	patience=7, verbose=0,
	mode='auto',
	restore_best_weights=True)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
 factor=0.1, patience=5, verbose=0, mode='auto',
 min_delta=0.0001, cooldown=0, min_lr=0)

######### train model ############
model.compile(optimizers.Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=num_train_samples // batch_size,
    epochs=num_epochs,
    callbacks=[reduce_lr,early_stop,tensorboard],
    validation_data=val_generator,
    validation_steps=num_val_samples // batch_size)


model.save_weights('first_try.h5')
