import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import os
from keras.applications
model = tf.keras.applications.ResNet51(include_top=False,weights='imagenet')

# transfer learning
for i in model.layers:
  i.trainable = False

global_avg = tf.keras.layers.GlobalAveragePooling2D()(model.output)
drop_out = tf.keras.layers.Dropout(0.4)(global_avg)
out = tf.keras.layers.Dense(2,activation='sigmoid')(drop_out)
resnet = tf.keras.Model(inputs=[model.input],outputs=[out]) import ResNet50
from keras.models import Sequential
from keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

### Data cleaning and organization
df = pd.read_csv('../datasets/HAM/HAM10000_metadata.csv')
print(df.head())

def append_ext(fn):
    return fn+".jpg"

df['image_id'] = df['image_id'].apply(append_ext)

### Organizing HAM Dataset

# base_dir = '../datasets/HAM/'
 
# class_buckets = [0] * 7

# class_dict = {
#     "akiec": 0,
#     "bcc": 1,
#     "bkl": 2,
#     "df": 3,
#     "mel": 4,
#     "nv": 5,
#     "vasc": 6
# }

# for elm in df['dx']:
#     if elm in class_dict:
#         class_buckets[class_dict[elm]] += 1

# for i in range(7):
#     print(class_buckets[i])

# for elm in df['image_id']:
#     disease_class = df[df['image_id']==elm]['dx'].iloc[0]
#     print(base_dir)
#     print(elm)
#     print(disease_class)
#     os.rename(base_dir + elm,'../datasets/HAM/' + disease_class + '/' + elm)

### Custom Model

# model = Sequential()
# model.add(Conv2D(8,(11, 11), activation='relu', input_shape=(600,450,3)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(16,(7, 7), activation='relu'))
# model.add(Conv2D(16,(7, 7), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(32, (5, 5), activation='relu'))
# model.add(Conv2D(32, (5, 5), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(16, (5, 5), activation='relu'))
# model.add(Conv2D(16, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(8, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(7, activation='softmax'))

### Pre-trained Model w/ Transfer Learning

model = tf.keras.applications.ResNet51(include_top=False,weights='imagenet')

# transfer learning
for i in model.layers:
  i.trainable = False

global_avg = tf.keras.layers.GlobalAveragePooling2D()(model.output)
drop_out = tf.keras.layers.Dropout(0.4)(global_avg)
out = tf.keras.layers.Dense(7,activation='sigmoid')(drop_out)
resnet = tf.keras.Model(inputs=[model.input],outputs=[out])


model.compile(
  optimizer='adam', 
  loss='categorical_crossentropy', 
  metrics=['accuracy']
)

model.summary()

epoch_count = 8
batch_size = 16

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

# split = int(df.shape[0]*0.8)

# train_generator = train_datagen.flow_from_dataframe(
#     dataframe=df[:split],
#     directory=base_dir,
#     x_col='image_id',
#     y_col='dx',
#     #batch_size=1848,
#     shuffle=True,
#     class_mode="categorical",
#     classes=['akiec','bcc','bkl','df','mel','nv','vasc'],
#     target_size=(600,450)
# )

# test_generator = test_datagen.flow_from_dataframe(
#     dataframe=df[split:],
#     directory=base_dir,
#     x_col='image_id',
#     y_col='dx',
#     #batch_size=1848,
#     shuffle=True,
#     class_mode="categorical",
#     classes=['akiec','bcc','bkl','df','mel','nv','vasc'],
#     target_size=(600,450)
# )

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

model.fit_generator(
	train_generator,
    steps_per_epoch=20,
    epochs=epoch_count,
    verbose=1,  
    validation_data=validation_generator,)

# Saving the model
model.save_weights('classify_v2.h5'.format(epoch_count))

# test_pred = pd.DataFrame(model.predict(test_datagen))
# test_pred = pd.DataFrame(test_pred.idxmax(axis = 1))
# test_pred.index.name = 'ImageId'
# test_pred = test_pred.rename(columns = {0: 'Label'}).reset_index()
# test_pred['ImageId'] = test_pred['ImageId'] + 1