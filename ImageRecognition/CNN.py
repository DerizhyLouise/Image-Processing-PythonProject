# Deep Learning CNN model to recognize object
'''This script uses a database of images and creates CNN model on top of it to test
   if the given image is recognized correctly or not'''

'''####### IMAGE PRE-PROCESSING for TRAINING and TESTING data #######'''

# Specifying the folder where images are present
TrainingImagePath='c:\Coding\Python\ImageRecognition\CNN\datatrain'

from keras.preprocessing.image import ImageDataGenerator
# Defining pre-processing transformations on raw images of training data
# These hyper parameters helps to generate slightly twisted versions
# of the original image, which leads to a better model, since it learns
# on the good and bad mix of images
train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

# Defining pre-processing transformations on raw images of testing data
# No transformations are done on the testing images
test_datagen = ImageDataGenerator()

# Generating the Training Data
training_set = train_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


# Generating the Testing Data
test_set = test_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

# Printing class labels for each object
test_set.class_indices

'''############ Creating lookup table for all images ############'''
# class_indices have the numeric tag for each object
TrainClasses=training_set.class_indices

# Storing the object and the numeric tag for future reference
ResultMap={}
for objectValue,objectName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[objectValue]=objectName

# Saving the object map for future reference
import pickle
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)

# The model will give answer as a numeric tag
# This mapping will help to get the corresponding object name for it
print("Mapping of Object and its ID",ResultMap)

# The number of neurons for the output layer is equal to the number of objects
OutputNeurons=len(ResultMap)
print('\n The Number of output neurons: ', OutputNeurons)

'''############ Creating lookup table for all objects ############'''
# class_indices have the numeric tag for each object
TrainClasses=training_set.class_indices

# Storing the object and the numeric tag for future reference
ResultMap={}
for objectValue,objectName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[objectValue]=objectName

# Saving the object map for future reference
import pickle
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)

# The model will give answer as a numeric tag
# This mapping will help to get the corresponding object name for it
print("Mapping of Object and its ID",ResultMap)

# The number of neurons for the output layer is equal to the number of objects
OutputNeurons=len(ResultMap)
print('\n The Number of output neurons: ', OutputNeurons)

'''######################## Create CNN deep learning model ########################'''
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

'''Initializing the Convolutional Neural Network'''
classifier= Sequential()

''' STEP--1 Convolution
# Adding the first layer of CNN
# we are using the format (64,64,3) because we are using TensorFlow backend
# It means 3 matrix of size (64X64) pixels representing Red, Green and Blue components of pixels
'''
classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'))

'''# STEP--2 MAX Pooling'''
classifier.add(MaxPool2D(pool_size=(2,2)))

'''############## ADDITIONAL LAYER of CONVOLUTION for better accuracy #################'''
classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))

classifier.add(MaxPool2D(pool_size=(2,2)))

'''# STEP--3 FLattening'''
classifier.add(Flatten())

'''# STEP--4 Fully Connected Neural Network'''
classifier.add(Dense(64, activation='relu'))

classifier.add(Dense(OutputNeurons, activation='softmax'))

'''# Compiling the CNN'''
#classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])

###########################################################
import time
# Measuring the time taken by the model to train
StartTime=time.time()

# Starting the model training
classifier.fit(
        x=training_set,
        steps_per_epoch=30,
        epochs=10,
        validation_data=test_set,
        validation_steps=10
)

EndTime=time.time()
print("###### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ######')

'''########### Making single predictions ###########'''
import numpy as np
import keras.utils as image

ImagePath='c:\Coding\Python\ImageRecognition\CNN\datatest\datatest2.jpg'
test_image=image.load_img(ImagePath,target_size=(64, 64))
test_image=image.img_to_array(test_image)

test_image=np.expand_dims(test_image,axis=0)

result=classifier.predict(test_image,verbose=0)
#print(training_set.class_indices)

print('####'*10)
print('Prediction is: ',ResultMap[np.argmax(result)])