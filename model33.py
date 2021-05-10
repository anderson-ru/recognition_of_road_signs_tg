import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import time
import warnings
warnings.filterwarnings("ignore")

# time the model with module 'time'
start = time.time()
# input_path
current_path = os.getcwd()


image_data = []
image_labels = []

# number of classes
classes = 43

# image format
height = 32
width = 32
channels = 3

# Load the images from the  path
for i in range(classes):
    path = current_path + "/Train/" + str(i)
    images = os.listdir(path)

    for img in images:
        try:
            image = cv2.imread(path + '/' + img)
            image_from_array = Image.fromarray(image, "RGB")
            resize_image = image_from_array.resize((height, width))
            image_data.append(np.array(resize_image))
            image_labels.append(i)
        except():
            print("Error in Image loading")


# Converting lists into numpy arrays
image_data = np.array(image_data)
image_labels = np.array(image_labels)

# Time taken to load our images in seconds
end = time.time()
print("Time taken to load images: ", round(end - start, 5), "seconds")

# Shuffling data
shuffle_indexes = np.arange(image_data.shape[0])
np.random.shuffle(shuffle_indexes)

image_data = image_data[shuffle_indexes]
image_labels = image_labels[shuffle_indexes]

# Splitting training and testing dataset
X_train, X_valid, y_train, y_valid = train_test_split(image_data, image_labels, test_size=0.2,
                                                      random_state=2666, shuffle=True)

# Scale the values between 0 and 1
X_train = X_train / 255
X_valid = X_valid / 255

# The dimensions concur
print("X_train.shape", X_train.shape)
print("X_valid.shape", X_valid.shape)
print("y_train.shape", y_train.shape)
print("y_valid.shape", y_valid.shape)

# Converting the labels into one hot encoding
y_train = keras.utils.to_categorical(y_train, classes)
y_valid = keras.utils.to_categorical(y_valid, classes)

# The dimensions concur
print(y_train.shape)
print(y_valid.shape)

# Clearing previous session if there was any
keras.backend.clear_session()
np.random.seed(2666)

# Create the model LeNet with Keras
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=18, kernel_size=(5,5), strides=1, activation="relu",
                        input_shape=(height, width, channels)),
    keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),
    keras.layers.Conv2D(filters=36, kernel_size=(5,5), strides=1, activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),
    keras.layers.Conv2D(filters=36, kernel_size=(5, 5), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(1,1)),
    keras.layers.Flatten(),
    keras.layers.Dense(units=72, activation="relu"),
    keras.layers.Dense(units=43, activation="softmax"),
])

model.summary()

# Compilation of our model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

epochs = 20
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_valid, y_valid))
validation_data = (X_valid, y_valid)


# Visualization the results
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


model.save('model33.h5')

# Testing accuracy on the reserved test set
test = pd.read_csv(current_path + "/Test.csv")

labels = test["ClassId"].values
test_imgs = test["Path"].values

# How an image looks like
img_index = 25
image = Image.open(current_path + '/' + test_imgs[img_index])
img = image.resize((height, width))
img = np.array(img) / 255.
img = img.reshape(1, height, width, channels)

print(img.shape)
print(labels[img_index])
plt.imshow(image)

# Dictionary to map classes.
classes = {
    0:'Speed limit (20km/h)',
    1:'Speed limit (30km/h)',
    2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)',
    4:'Speed limit (70km/h)',
    5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)',
    7:'Speed limit (100km/h)',
    8:'Speed limit (120km/h)',
    9:'No passing',
    10:'No passing veh over 3.5 tons',
    11:'Right-of-way at intersection',
    12:'Priority road',
    13:'Yield',
    14:'Stop',
    15:'No vehicles',
    16:'Veh > 3.5 tons prohibited',
    17:'No entry',
    18:'General caution',
    19:'Dangerous curve left',
    20:'Dangerous curve right',
    21:'Double curve',
    22:'Bumpy road',
    23:'Slippery road',
    24:'Road narrows on the right',
    25:'Road work',
    26:'Traffic signals',
    27:'Pedestrians',
    28:'Children crossing',
    29:'Bicycles crossing',
    30:'Beware of ice/snow',
    31:'Wild animals crossing',
    32:'End speed + passing limits',
    33:'Turn right ahead',
    34:'Turn left ahead',
    35:'Ahead only',
    36:'Go straight or right',
    37:'Go straight or left',
    38:'Keep right',
    39:'Keep left',
    40:'Roundabout mandatory',
    41:'End of no passing',
    42:'End no passing veh > 3.5 tons'
          }

# Prediction of this image
pred = model.predict_classes(img)[0]
print(pred)

sign = classes[pred]
print(sign)

# Load and preprocessed test set
start = time.time()
test = pd.read_csv(current_path + '/Test.csv')

labels = test["ClassId"].values
imgs = test["Path"].values

data = []

for img in imgs:
    try:
        image = cv2.imread(current_path + '/' + img)
        image_from_array = Image.fromarray(image, 'RGB')
        resize_image = image_from_array.resize((height, width))
        data.append(np.array(resize_image))
    except():
        print("Error")

X_test = np.array(data)
X_test = X_test / 255

# Prediction of test set
pred = model.predict_classes(X_test)

# Accuracy with the test data
print("Accuracy score: ", accuracy_score(labels, pred))
end = time.time()
print("Time taken: ", round(end-start, 5), "seconds")
