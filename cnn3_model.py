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
from PIL import Image
import matplotlib.pyplot as plt
import random
import glob
import time
from scipy import stats
import keras
from keras.utils.vis_utils import plot_model

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
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(nrows, ncolumns, channels)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(drop_rate))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(drop_rate))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(drop_rate))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(drop_rate))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(drop_rate))  #Dropout for regularization
model.add(layers.Dense(128 , activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.summary()

 
# Compile the model (we can play with the optimizer)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

               

## IF NEEDED we perform data augmentation
# Set the transformations and augmentations for the training set (in cases of small datasets this helps)
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,#Scale the image between 0 and 1
                                    vertical_flip=True,
                                    horizontal_flip=True,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.1)
                                    # rotation_range=90,

# Set the validation dataset but not augmenting it (only rescaling it)
val_datagen = ImageDataGenerator(rescale=1./255)
# Set the test dataset but not augmenting it (only rescaling it)
test_datagen = ImageDataGenerator(rescale=1./255)
# Create the generators
 
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
 
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)
 
test_generator = test_datagen.flow(X_test, y_test, batch_size=batch_size, shuffle=False)
# Model training
start = time.time()
history = model.fit(train_generator,
                                steps_per_epoch = ntrain // batch_size,
                                epochs=num_epochs,
                                validation_data = val_generator,
                                validation_steps = ntest // batch_size)

end = time.time()
print('TIME = ', end-start, "sec.")
model.save("rgb")
 
# Model evaluation
val_generator.reset() # reset the generator to force it start from the begining
eval_gen = model.evaluate(val_generator, steps = nval // batch_size, workers=1, use_multiprocessing=False)
print("*****************************")
print("Evaluation accuracy and loss")
print(" accuracy =", eval_gen[1] * 100)
print(" loss =", eval_gen[0])
print("*****************************")
test_generator.reset() # reset the generator to force it start from the begining
pred_gen = model.evaluate(test_generator, steps = ntest // batch_size, workers=1, use_multiprocessing=False)
print("Testing accuracy and loss")
print(" accuracy =", pred_gen[1] * 100)
print(" loss =", pred_gen[0])
print("*****************************")
 
test_generator.reset() # reset the generator to force it start from the begining

predictions = model.predict(test_generator, steps =None, workers=1, use_multiprocessing=False)
 
predictions = predictions > 0.5
 

 
 
# Separate Confusion Matrices
y_true = y_test
y_pred = predictions
 
labels = ['infarction', 'Ischemia', 'Normal']
 
conf_mat_dict={}
from sklearn.metrics import confusion_matrix
for label_col in range(len(labels)):
    y_true_label = y_true[:, label_col]
    y_pred_label = y_pred[:, label_col]
    conf_mat_dict[labels[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)
 
 
for label, matrix in conf_mat_dict.items():
    print("Confusion matrix for label {}:".format(label))
    print(matrix)
 

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
labels = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
cm_gen = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))#, labels=labels)
print(cm_gen)
 

 
target_names = ['infarction', 'Ischemia', 'Normal']
from sklearn.metrics import classification_report
cr = classification_report(y_test, predictions, target_names=target_names)
print(cr)
 
 
from sklearn.metrics import roc_auc_score
print("####")

from sklearn import  metrics



print(metrics.roc_auc_score(y_test, predictions))


#Plot accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


from sklearn.metrics import roc_curve, auc
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(classes_num):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(classes_num):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
