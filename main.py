# Import necessary libraries and packets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import seaborn as sns
import random as rd
import keras
from sklearn.metrics import confusion_matrix
from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D

# Read the data from .csv file by pass the file path as a parameter
train_dataset_path = pd.read_csv('/Users/macbookpro/Desktop/SignLanguageMNIST/dataset/sign_mnist_train.csv')
test_dataset_path = pd.read_csv('/Users/macbookpro/Desktop/SignLanguageMNIST/dataset/sign_mnist_test.csv')

# Convert data to array-like
train_data = np.array(train_dataset_path)
test_data = np.array(test_dataset_path)
# Get all data from pixel1, pixel2,..., pixel784
x_train = train_data[:, 1:]
x_test = test_data[:, 1:]
# Get all the label from each image, we will use it to compare the prediction with the initial label
y_train = train_data[:, 0]
y_test = test_data[:, 0]
# Reshape x_train, x_test from (1 x 784) array to (28 x 28) array
x_train = x_train.reshape(x_train.shape[0], *(28, 28, 1))
x_test = x_test.reshape(x_test.shape[0], *(28, 28, 1))

# Training dataset overview
fig = plt.figure(figsize=(12, 9))
axes = sns.countplot(x="label", data= train_dataset_path)
axes.set_ylabel('Count')
axes.set_title('Label of images im train dataset')
plt.show()

# Testing dataset overview
fig = plt.figure(figsize=(12, 9))
axes = sns.countplot(x="label", data= test_dataset_path)
axes.set_ylabel('Count')
axes.set_title('Label of images in test dataset')
plt.show()

plt.figure(figsize=(18, 9))
for i in range(21):
    plt.subplot(3, 7, i + 1)
    plt.imshow(x_train[i])
plt.show()

# Scale value of x_train, x_test from [0, 255] -> [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0
# 21 sample image, grayscale
plt.figure(figsize=(18, 9))
for i in range(21):
    plt.subplot(3, 7, i + 1)
    plt.imshow(x_train[i], cmap= 'gray')
plt.show()

# Build Convolution Neutral Network Model
model = keras.Sequential()
model.add(keras.Input(shape= (28, 28, 1)))
model.add(keras.layers.Conv2D(filters= 32, kernel_size= (3, 3), activation= 'relu'))
model.add(keras.layers.MaxPool2D(pool_size= (2, 2)))
model.add(keras.layers.Dropout(rate= 0.3))
model.add(keras.layers.Conv2D(filters= 64, kernel_size= (3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size= (2, 2)))
model.add(keras.layers.Dropout(rate= 0.3))
model.add(keras.layers.Conv2D(filters= 128, kernel_size= (3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size= (2, 2)))
model.add(keras.layers.Dropout(rate= 0.3))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units= 128, activation= 'relu'))
model.add(keras.layers.Dense(units= 25, activation= 'softmax'))
# Configures the model for training.
model.compile(loss= 'sparse_categorical_crossentropy', optimizer= 'adam', metrics= ['acc'])
# Reduce learning rate when a metric has stopped improving.
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience= 2, verbose= 1, mode='max', min_lr= 0.00001)
# Prints a string summary of the network.
model.summary()

# Trains the model for a fixed number of epochs (dataset iterations)
EPOCHS = 30
BATCH_SIZE = 128
start_time = time.time()
history = model.fit(x_train, y_train, batch_size= BATCH_SIZE, epochs= EPOCHS, verbose= 1, validation_data= (x_test, y_test))
end_time = time.time()
print(f'Elapsed time = {round(end_time - start_time, 2)} seconds.')

loss = history.history['loss']
validation_loss = history.history['val_loss']
epochs_range = range(1, len(loss) + 1)
plt.plot(epochs_range, loss, 'b', label= 'Training loss')
plt.plot(epochs_range, validation_loss, 'g', label= 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

accurancy = history.history['acc']
val_accurancy = history.history['val_acc']
plt.plot(epochs_range, accurancy, 'b', label= 'Training accurancy')
plt.plot(epochs_range, val_accurancy, 'g', label= 'Validation accurancy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Make some first predictions
plt.figure(figsize=(18, 9))
for i in range(21):
    plt.subplot(3, 7, i + 1)
    testImage = x_test[i]
    prediction = model.predict(testImage.reshape(-1,28,28,1))
    plt.imshow(testImage.reshape(28, 28), cmap = 'gray')
    plt.xlabel(f"Prediction:{np.argmax(prediction)} \n Actual Value:{y_test[i]}")
plt.show()

# Create a list of predictions for all images of test data
predictions_list = []
for i in range(len(y_test)):
    testImage = x_test[i]
    prediction = model.predict(testImage.reshape(-1,28,28,1))
    predictions_list.append(np.argmax(prediction))
# Create and plot the confusion matrix
CONFUSION_MATRIX = confusion_matrix(y_test, predictions_list)
fig, axes = plt.subplots(figsize= (18, 9))
sns.set_theme(font_scale= 1)
sns.heatmap(CONFUSION_MATRIX, annot= True, linewidths= 0.5, ax= axes)
plt.show()

# Return the loss value and accurancy value for the model in test mode.
loss, acc = model.evaluate(x_test, y_test)
print(f"Loss: {loss}, Accuracy: {acc}")