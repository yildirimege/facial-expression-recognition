import os
import numpy as np
import cv2 as cv
import keras
from keras.utils import np_utils

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


data_path = "CK+48"
data_list = os.listdir(data_path)
img_data_list = []
for dataset in data_list:
    img_list = os.listdir(data_path + '/' + dataset)         #Loading the dataset from file system
    print(f'loaded the {dataset} images \n')
    for image in img_list:
        input_img = cv.imread(data_path + '/' + dataset + '/' + image)
        img_data_list.append(input_img)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data / 255  # Scaling down / normalizing the pixels

num_classes = 7  # Label count aka how many expressions
num_samples = img_data.shape[0]  # Number of total images. 980 in CK+ case
labels = np.ones((num_samples,), dtype='int64')

labels[0:134] = 0  # 135             #Labeling each image in the order of files.
labels[135:188] = 1  # 54
labels[189:365] = 2  # 177
labels[366:440] = 3  # 75
labels[441:647] = 4  # 207
labels[648:731] = 5  # 84
labels[732:980] = 6  # 249

expression_names = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'] #Naming each label.

#
Y = np_utils.to_categorical(labels, num_classes=num_classes)
x, y = shuffle(img_data, Y, random_state=2) #Shuffles images to do random permutations, random number generating for shuffling is enabled.

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.15, random_state=2) #Splitting dataset for train and test since validation data is not seperatly supplied


input_shape = (48, 48, 3) #All input pictures are 48x48 pixels wide

#I chose small convolution filter values since even the raw image has many valuable informations that can be extracted.
# But i followed "Test driven programming" techniques, i am sure pooling and filtering will be improved when being sure of those values.
# Also my model is clearly overfitting. I'll make sure it doesnt before startıng with a different dataset.


model = keras.Sequential()
model.add(keras.layers.Conv2D(8, (5, 5), input_shape=input_shape, padding='same', activation='relu')) #Input Layer
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(16, (5, 5), padding='same', activation = 'relu')) #Hidden Layer 1
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(64, (3, 3), activation = 'relu'))     #Hidden Layer 2
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation = 'relu'))             #Output Layer
model.add(keras.layers.Dense(num_classes, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')
#Crossentropy loss function is good on classification algorithms when the output layer is a probability (softmax in this case)

model.summary() #Printing summary of model ( Layers, pooling methods and shapes etc.

history = model.fit(X_train, y_train, batch_size=10, epochs=40, verbose=1, validation_data=(X_test, y_test)) #Training the model
