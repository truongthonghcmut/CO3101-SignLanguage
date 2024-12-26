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

model = keras.models.load_model("cnn_model.keras")


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


counter = 0
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

    if key % 256 == 32:
        img_name = "opencv_frame_{}.png".format(counter)
        cv2.imwrite(img_name, roi)
        print("{} written!".format(img_name))
        counter += 1

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()