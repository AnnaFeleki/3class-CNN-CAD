import matplotlib.pyplot as plt
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import glob
import time
import keras

def read_and_process_image():
  ...
  
  
  
# Select number of channels
channels = 3  # 1 grayscale, 3 rgb
# Create the X, y datasets for the ML processing
X,y = read_and_process_image(addrs)
 
X = np.array(X)
y = np.array(y)
 
print(X.shape)
 

from keras import utils as np_utils
y = keras.utils.np_utils.to_categorical(y, num_classes=classes_num)
print(y.shape)

 
# Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
 
# Get the length of the train and validation data
ntrain = len(X_train)
nval = len(X_val)
ntest = len(X_test)
print("X train length:", len(X_train))
print("X validation length:", len(X_val))
print("X test length:", len(X_test))
 
# Import CNN-related libraries
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
 
 
# Batch size (should be a factor of 2.***4,8,16,32,64...***)
batch_size = 32
# Dropout rate
drop_rate = 0.2
# Number of epochs
num_epochs = 500
 
# Initiate and build the model
model = Sequential()
....

