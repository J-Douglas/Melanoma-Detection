import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

### Datasets
df_train = pd.read_csv('../datasets/ISIC 2020/train.csv')
print(df_train.head())

df_test = pd.read_csv('../datasets/ISIC 2020/test.csv')

def append_ext(fn):
    return fn+".jpg"

df_train['image_name'] = df_train['image_name'].apply(append_ext)
df_test['image_name'] = df_test['image_name'].apply(append_ext)

### Custom Model

model = Sequential()
model.add(Conv2D(32,(5,5), activation='relu', input_shape=(6000,4000,3)))
model.add(Conv2D(64,(5,5), activation='relu'))
model.add(Conv2D(64,(5,5), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Conv2D(32,(3,3), activation='relu'))
model.add(Conv2D(16,(3,3), activation='relu'))
model.add(Conv2D(8,(3,3), activation='relu'))
model.add(Dense(2, activation='softmax'))

### Pre-trained Model w/ Transfer Learning

