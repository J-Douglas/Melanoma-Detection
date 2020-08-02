import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical

df = pd.read_csv('../datasets/HAM/HAM10000_metadata.csv')

train_img = pd.read_csv('../datasets/HAM/')
train_labels
test_img
test_labels

model = Sequential()