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
EPOCHS = 50
BATCH_SIZE = 128
start_time = time.time()
history = model.fit(x_train, y_train, batch_size= BATCH_SIZE, epochs= EPOCHS, verbose= 1, validation_data= (x_test, y_test))
end_time = time.time()
print(f'Elapsed time = {round(end_time - start_time, 2)} seconds.')

# Return the loss value and accurancy value for the model in test mode.
loss, acc = model.evaluate(x_test, y_test)
print(f"Loss: {loss}, Accuracy: {acc}")

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


import cv2

def predict_sign(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image.reshape(-1, 28, 28, 1) / 255.0
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)
    return predicted_label


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press P to make a prediction. Press Q to exit the program")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive image from camera.")
        break

    x, y, w, h = 50, 50, 400, 400
    roi = frame[y:y+h, x:x+w]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        predicted_label = predict_sign(roi, model)
        print(f"Pre: {predicted_label}")

    cv2.imshow("Sign Language Recognition", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()