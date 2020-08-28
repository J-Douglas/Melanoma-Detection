import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50

### Datasets
df_train = pd.read_csv('../datasets/ISIC 2020/train.csv')
print(df_train.head())

df_test = pd.read_csv('../datasets/ISIC 2020/test.csv')

def append_ext(fn):
    return fn+".jpg"

df_train['image_name'] = df_train['image_name'].apply(append_ext)
df_test['image_name'] = df_test['image_name'].apply(append_ext)

### Organizing ISIC 2020 Dataset

# base_dir = '../datasets/ISIC 2020/'

# for elm in df_train['image_name']:
#     target = df_train[df_train['image_name']==elm]['benign_malignant'].iloc[0]
#     print(base_dir)
#     print(elm)
#     print(target)
#     os.rename(base_dir + 'train/' + elm,'../datasets/ISIC 2020/' + target + '/' + elm)

# for elm in df_test['image_name']:
#     target = df_test[df_test['image_name']==elm]['benign_malignant'].iloc[0]
#     print(base_dir)
#     print(elm)
#     print(target)
#     os.rename(base_dir + 'test/' + elm,'../datasets/ISIC 2020/' + target + '/' + elm)

### Image Generators (train and testing data)

train_datagen = ImageDataGenerator(
    width_shift_range=0.3,
    height_shift_range=0.3,
    fill_mode="nearest",
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255)

validation_datagen = ImageDataGenerator(
    width_shift_range=0.3,
    height_shift_range=0.3,
    fill_mode="nearest",
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255)

test_datagen = ImageDataGenerator(
    width_shift_range=0.3,
    height_shift_range=0.3,
    fill_mode="nearest",
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        batch_size=batch_size,
		directory='../datasets/HAM/train',
        target_size=(600, 450), 
        classes = ['akiec','bcc','bkl','df','mel','nv','vasc'],
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        batch_size=batch_size,
        directory='../datasets/HAM/validation',
        target_size=(600, 450), 
        classes = ['akiec','bcc','bkl','df','mel','nv','vasc'],
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
		batch_size=batch_size,
        directory='../datasets/HAM/test',
        target_size=(600, 450), 
        classes = ['akiec','bcc','bkl','df','mel','nv','vasc'],
        class_mode='categorical')

### Custom Model

# model = Sequential()
# model.add(Conv2D(32,(5,5), activation='relu', input_shape=(6000,4000,3)))
# model.add(Conv2D(64,(5,5), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64,(5,5), activation='relu'))
# model.add(Conv2D(64,(3,3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64,(3,3), activation='relu'))
# model.add(Conv2D(32,(3,3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(16,(3,3), activation='relu'))
# model.add(Conv2D(8,(3,3), activation='relu'))
# model.add(Dense(2, activation='softmax'))

### Pre-trained Model w/ Transfer Learning

model = tf.keras.applications.ResNet51(include_top=False,weights='imagenet')

# transfer learning
for i in model.layers:
  i.trainable = False

global_avg = tf.keras.layers.GlobalAveragePooling2D()(model.output)
drop_out = tf.keras.layers.Dropout(0.4)(global_avg)
out = tf.keras.layers.Dense(2,activation='sigmoid')(drop_out)
resnet = tf.keras.Model(inputs=[model.input],outputs=[out])

resnet.compile(optimizer=tf.keras.optimizers.Adam(),loss="binary_crossentropy",metrics=["accuracy"])

epochs = 10

history = nasnet.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)


