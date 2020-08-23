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