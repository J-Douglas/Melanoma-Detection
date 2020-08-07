import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

### Data cleaning and organization
df = pd.read_csv('../datasets/HAM/HAM10000_metadata.csv')

base_dir = '../datasets/HAM/HAM10016_images/'

# for img in df[]:
# 	if elm :


model = Sequential()
model.add(Conv2D(32,(7, 7), activation='relu', input_shape=(600,450,3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(7, activation='softmax'))

model.compile(
  optimizer='adam', 
  loss='categorical_crossentropy', 
  metrics=['accuracy']
)

model.summary()

epoch_count = 20
batch_size = 120

### Image Generators (train and testing data)
train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

split = int(df.shape[0]*0.8)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df[:split],
    directory=base_dir,
    x_col='image_id',
    y_col='dx',
    #batch_size=1848,
    shuffle=False,
    class_mode="categorical",
    classes=['akiec','bcc','bkl','df','mel','nv','vasc'],
    target_size=(600,450)
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df[split:],
    directory=base_dir,
    x_col='image_id',
    y_col='dx',
    #batch_size=1848,
    shuffle=False,
    class_mode="categorical",
    classes=['akiec','bcc','bkl','df','mel','nv','vasc'],
    target_size=(600,450)
)

# train_generator = train_datagen.flow_from_directory(
# 		'../datasets/train',
#         target_size=(600, 450), 
#         batch_size=batch_size,
#         classes = ['akiec','bcc','bkl','df','mel','nv','vasc'],
#         class_mode='categorical')

# test_generator = test_datagen.flow_from_directory(
# 		'../datasets/test',
#         target_size=(600, 450), 
#         batch_size=batch_size,
#         classes = ['akiec','bcc','bkl','df','mel','nv','vasc'],
#         class_mode='categorical')

model.fit(
	train_generator, 
    steps_per_epoch=int(df.shape[0]/batch_size),  
    epochs=epoch_count,
    verbose=1)

# Saving the model
model.save_weights('skin_disease_classification.h5'.format(epoch_count))

test_pred = pd.DataFrame(model.predict(test_datagen))
# test_pred = pd.DataFrame(test_pred.idxmax(axis = 1))
# test_pred.index.name = 'ImageId'
# test_pred = test_pred.rename(columns = {0: 'Label'}).reset_index()
# test_pred['ImageId'] = test_pred['ImageId'] + 1