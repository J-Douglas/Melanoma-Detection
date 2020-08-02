import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical

df_labels = pd.read_csv('../datasets/HAM/HAM10000_metadata.csv')

train_img = df.iloc[]
train_labels =
test_img
test_labels

model = Sequential()
model.add(Conv2D(32,(5,5), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(6, activation='softmax'))

model.compile(
  optimizer='adam', 
  loss='categorical_crossentropy', 
  metrics=['accuracy']
)

model.summary()

epoch_count = 20
batch_count = 120

model.fit(
	train_img,
	train_labels,
	epochs=epoch_count,
	batch_size=batch_count
	)

# Saving the model
model.save_weights('skin_disease_classification.h5'.format(epoch_count))

test_pred = pd.DataFrame(model.predict(test_img, batch_size=batch_count))
test_pred = pd.DataFrame(test_pred.idxmax(axis = 1))
test_pred.index.name = 'ImageId'
test_pred = test_pred.rename(columns = {0: 'Label'}).reset_index()
test_pred['ImageId'] = test_pred['ImageId'] + 1