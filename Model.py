import os
import urllib2
import numpy as np
from PIL import Image
from cv2 import resize
from keras.models import Sequential
from keras.preprocessing import image
from vgg16_places_365 import VGG16_Places365
from keras.layers import Dense, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt
import collections
import keras.callbacks


# pre-trained weights
VGG_weights_path = "/home/chris/Downloads/vgg16-places.h5"

#pretrained network works with this image size
img_width = 224
img_height = 224

num_classes = 15

train_data_dir = "/home/chris/Desktop/CW_data/training/training"
val_data_dir = "/home/chris/Desktop/CW_data/validation"

# training parameters
batch_size = 10
num_train_samples = num_classes * 80 
num_val_samples = num_classes * 20 
num_epochs = 10000
augment_data_dir = "/home/chris/Desktop/CW_data/augmented_data"

#create dictionary to get class labels
class_names = ["bedroom", "Coast", "Forest", "Highway", "industrial", "Insidecity", "kitchen", "livingroom", "Mountain", "Office", "OpenCountry", "store", "Street", "Suburb", "TallBuilding"]
class_names = sorted(class_names)
id_to_class_name = dict(zip(range(len(class_names)), class_names))


weights_path = "/home/chris/Desktop/Scene_Recognition_Python/weights.h5"
test_data_dir = "/home/chris/Desktop/CW_data/testing"


def create_model():
    # pretrained load model
    vgg16 = VGG16_Places365(weights_path=VGG_weights_path)
    model = Sequential()
    # add all layers except the last one that does classification
    for layer in vgg16.layers[:-1]:
        model.add(layer)

    #freeze all the layers before the last (5th) convolutional block
    for layer in model.layers:
        if "block5" in layer.name:
            break
        else:
            layer.trainable = False;

    # add final dense prediction layer
    model.add(Dense(num_classes, activation='softmax', name="predictions"))
    model.summary()
    return model

def train_model():
    #augment data
    augment_data_gen = ImageDataGenerator(
        data_format='channels_last',
        brightness_range=[0.5,1.5],
        # width_shift_range=0.2, 
        # height_shift_range=0.2, 
        # rescale=1. / 255,
        # fill_mode='nearest',
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )
    #default conversion to rgb
    train_generator = augment_data_gen.flow_from_directory(
            train_data_dir,
            shuffle=True,
            target_size=(img_width, img_height), 
            batch_size=batch_size,
            class_mode='categorical')

    #for validation only do rescaling as augmentation
    simple_data_gen = ImageDataGenerator()
    val_generator = simple_data_gen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    model = create_model()

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

    model.save_weights('weights_part_aug_shear.h5')


def load_treained_model(weights_path):
    model = create_model()
    model.load_weights(weights_path)
    return model

#return a dictionary from image ID to pixels 2d array
def load_images(test_data_dir):
    images = []
    image_ids = []
    for filename in os.listdir(test_data_dir):
        if filename.endswith(".jpg"):
            img = image.load_img(test_data_dir + "/" +filename, target_size=(224, 224))
            img_array = image.img_to_array(img)
            images.append(img_array)
            img_id = int(filename.split(".")[0])
            image_ids.append(img_id)

    image_id_to_pixels = dict(zip(image_ids, images))
    image_id_to_pixels = collections.OrderedDict(sorted(image_id_to_pixels.items()))
    return image_id_to_pixels


def predict(weights_path, test_data_dir):
    model = load_treained_model(weights_path)
    image_id_to_pixels = load_images(test_data_dir)
    
    #write predictions
    with open("run3.txt", "w") as text_file:
    	for img_id, pixels in image_id_to_pixels.iteritems():
            prediction = model.predict(np.expand_dims(pixels, axis=0))
            id = np.argmax(prediction[0])

            text_file.write(str(img_id) + ".jpg " + id_to_class_name[id] + "\n")
    


train_model()