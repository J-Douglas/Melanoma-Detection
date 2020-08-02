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

