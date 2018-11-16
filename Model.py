import os
import urllib2
import numpy as np
from PIL import Image
from cv2 import resize

from vgg16_places_365 import VGG16_Places365
'''
TEST_IMAGE_URL = 'http://places2.csail.mit.edu/imgs/demo/6.jpg'

image = Image.open(urllib2.urlopen(TEST_IMAGE_URL))
image = np.array(image, dtype=np.uint8)
image = resize(image, (224, 224))
image = np.expand_dims(image, 0)
'''
print("start")
model = VGG16_Places365(weights='places')
print("done")
'''
predictions_to_return = 5
preds = model.predict(image)[0]
top_preds = np.argsort(preds)[::-1][0:predictions_to_return]

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

print('--SCENE CATEGORIES:')
# output the prediction
for i in range(0, 5):
    print(classes[top_preds[i]])

# --PREDICTED SCENE CATEGORIES:
# cafeteria
# food_court
# restaurant_patio
# banquet_hall
# restaurant
'''