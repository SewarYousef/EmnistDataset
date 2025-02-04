# Import modules
import matplotlib.pyplot as plt
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import tensorflow as tf  # used to perform training sessions
# Keras
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix,classification_report
train = pd.read_csv(r"C:\Users\sewar\OneDrive\Desktop\datasets\emnist_data\emnist-balanced-train.csv")
test = pd.read_csv(r"C:\Users\sewar\OneDrive\Desktop\datasets\emnist_data\emnist-balanced-test.csv")
# A delimiter is one or more characters that separate text strings. When a program stores sequential or tabular data, it
# delimits each item of data with a predefined character

# Constants
HEIGHT = 28
WIDTH = 28

# Split x and y
train_x = train.iloc[:, 1:]
train_y = train.iloc[:, 0]
del train

test_x = test.iloc[:, 1:]
test_y = test.iloc[:, 0]
del test


def rotate(image):
    image = image.reshape([HEIGHT, WIDTH])
    return image


# Flip and rotate image
train_x = np.asarray(train_x)
train_x = np.apply_along_axis(rotate, 1, train_x)
test_x = np.asarray(test_x)
test_x = np.apply_along_axis(rotate, 1, test_x)

# Flatten Data
dims = train_x.shape[1] * train_x.shape[2]
X_train = train_x.reshape(train_x.shape[0], dims)
X_test = test_x.reshape(test_x.shape[0], dims)
print("train_x:", train_x.shape)
print("test_x:", test_x.shape)

# Normalizing the data
# Rescale to 0 -> 1 by dividing by max pixel value (255)
# Making sure that the values are float so that we can get decimal points after division
train_x = train_x.astype('float32') / 255
test_x = test_x.astype('float32') / 255

# Image after being normalized
plt.imshow(train_x.reshape(len(train_x), 28, 28)[55].T, cmap='gray')
plt.colorbar()
plt.show()

# One-Hot Encoding
from keras.utils import np_utils  # used to convert array of labeled data to one-hot vector
# Effects accuracy as have a class where their will be no results
num_classes = train_y.nunique()
print("number of classes", num_classes)
y_train = np_utils.to_categorical(train_y, num_classes)
y_test = np_utils.to_categorical(test_y, num_classes)

# Empty Sequential model
from tensorflow.keras.models import Sequential
model = Sequential()

# Layers
# 1 - number of elements (pixels) in each image
# Dense layer - when every node from previous layer is connected to each node in current layer
model.add(Dense(500, activation='relu'))

# Second Hidden Layer
model.add(Dense(500, activation='relu'))

# Output Layer - number of nodes corresponds to number of y labels
model.add(Dense(num_classes, activation='softmax'))

'''These are densely connected, or fully connected, neural layers. The first Dense layer has 128 nodes (or neurons). The
 second (and last) layer returns a logits array with length of 10. Each node contains a score that indicates the current 
 image belongs to one of the 10 classes'''

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, batch_size=128, epochs=10, shuffle=True, verbose=2)

# we can plot and see some visual content.
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.legend(['loss', 'accuracy'])
plt.title('Loss VS accuracy')
plt.xlabel('epoch')
plt.show()
print("model summary\n")
model.summary()

# Save Model
model.save("emnist_trained.h5")

# Load Model
from tensorflow.keras.models import load_model

model = load_model("emnist_trained.h5")

# Evaluate Model
model_loss, model_accuracy = model.evaluate(X_test, y_test, verbose= 2)
print(f"Loss: {model_loss}, accuracy: {model_accuracy}")


probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(X_test)
print(np.argmax(predictions[0]))

labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefjhijklmnopqrstuvwxyz'
import random
random.seed(1123)
sample = np.arange(test_x.shape[0])
np.random.shuffle(sample)
sample = sample[0:10]

# format from input into network
results = np.round(model.predict(X_test[sample], verbose=1), decimals=2)
resultLabels = np.argmax(results, axis=1)

fig = plt.figure(figsize=(15, 8))
for i in range(10):
    fig.add_subplot(2, 5, i+1, aspect='equal')
    plt.imshow(test_x[sample[i]].T, cmap='gray')
    plt.title('Class {}'.format(labels[resultLabels[i]]))
    plt.xlabel('Imag {}'.format(sample[i]))

plt.show()

plt.imshow(X_test[0].reshape(28, 28).T, cmap='gray')
plt.show()
plt.imshow(test_x[sample[0]].T, cmap='gray')
plt.title('Class {}'.format(resultLabels[0]))
plt.xlabel('Imag {}'.format(sample[0]))
# pred = model.predict(X_test[0].reshape(1, 28, 28, 1))
# print(pred.argmax())


# file = r"C:\Users\sewar\OneDrive\Desktop\shAI Club\emnist images\H2.PNG"
#
# # Convert to numpy array
# from tensorflow.keras.preprocessing import image
# image_size = (28,28)
# im = image.load_img(file, target_size=image_size, color_mode="grayscale")
#
# #scale and flatten
# from tensorflow.keras.preprocessing.image import img_to_array
# image = img_to_array(im)
#
# image /= 255
# img = image.flatten().reshape(-1, 28*28)
#
# img = 1 - img

# # Plot and Predict
# plt.imshow(img.reshape(28,28), cmap=plt.cm.Greys)
# model.predict_classes(img)
# image_size = (28, 28)
# im = image.load_img(file, target_size=image_size, color_mode="grayscale")

# import cv2
# image = cv2.imread(r"C:\Users\sewar\OneDrive\Desktop\shAI Club\emnist images\H.PNG")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# resized = cv2.resize(gray, (28,28))
# print(cv2.imwrite(r"C:\Users\sewar\OneDrive\Desktop\shAI Club\emnist images\H2.PNG", resized))

# import cv2
#
# img_array = np.asarray(img)
# resized = cv2.resize(img_array, (28, 28 ))
# gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) #(28, 28)
# image = cv2.bitwise_not(gray_scale)
#
# plt.imshow(image, cmap=plt.get_cmap('gray'))
# plt.show()
#
# image = image / 255
# image = image.reshape(1, 784)
#
# prediction = model.predict_classes(image)
# print("predicted digit or character:", str(prediction))
# plt.imshow(test_x[sample[image]].T, cmap='gray')
# plt.title('Class {}'.format(resultLabels[image]))
# plt.xlabel('Imag {}'.format(sample[image]))
#
# #
# labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefjhijklmnopqrstuvwxyz'
# import random
# random.seed(1123)
# sample = np.arange(test_x.shape[0])
# np.random.shuffle(sample)
# sample = sample[0:10]

# # format from input into network
# results = np.round(model.predict(X_test[sample], verbose=1), decimals=2)
# resultLabels = np.argmax(results, axis=1)
# print(resultLabels)


# # Predict Model
test = np.expand_dims(X_train[8], axis=0)
# print(test.shape())

from sklearn.preprocessing import MinMaxScaler
#
scaler = MinMaxScaler().fit(X_train)
#
plt.imshow(scaler.inverse_transform(test).reshape(28, 28), cmap=plt.cm.Greys)
plt.show()
# #
print(model.predict(test).round())
print(model.predict_classes(test))

test = np.expand_dims(X_train[22], axis=0)
print(test.shape)
#
plt.imshow(scaler.inverse_transform(test).reshape(28, 28), cmap=plt.cm.Greys)
plt.show()
#
print(model.predict(test).round())
print(model.predict_classes(test))

#-----------------------------
# Import Custom Image
# file = r"/Users/hayoom/Downloads/h.png"
#
# # Convert to numpy array
# from tensorflow.keras.preprocessing import image
# #
# image_size = (28, 28)
# img = image.load_img(file, target_size=image_size, color_mode="grayscale")
#
# # scale and flatten
# from tensorflow.keras.preprocessing.image import img_to_array
#
# image = img_to_array(img)
#
# image /= 255
# img = image.flatten().reshape(-1, 28 * 28)
#
# img = 1 - img
#
# # We will resize this image and greyscale it
# import cv2
#
# img_array = np.asarray(img)
# resized = cv2.resize(img_array, (28, 28 ))
# gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) #(28, 28)
# image = cv2.bitwise_not(gray_scale)
#
# plt.imshow(image, cmap=plt.get_cmap('gray'))
# print(image)
#
# # Now we will resize it and feed the input
# image = image / 255
# image = image.reshape(1, 784)
# prediction = model.predict_classes(image)
# print("predicted digit or character:", str(prediction))
#
#
# # Import Custom Image
# # Plot and Predict
# plt.imshow(img.reshape(28, 28), cmap=plt.cm.Greys)
# plt.show()
# print(model.predict_classes(img), '\n')
#
# file = r"C:\Users\sewar\OneDrive\Desktop\shAI Club\emnist images\H2.PNG"
#
# from tensorflow.keras.preprocessing import image
#
# image_size = (28, 28)
# im = image.load_img(file, target_size=image_size, color_mode="grayscale")
#
# from tensorflow.keras.preprocessing.image import img_to_array
#
# image = img_to_array(im)
#
# image /= 255
# img = image.flatten().reshape(-1, 28 * 28)
#
# img = 1 - img
# plt.imshow(img.reshape(28, 28), cmap=plt.cm.Greys)
# plt.show()
# print(model.predict_classes(img), '\n')
#
#
# import cv2
# image = cv2.imread(r"C:\Users\sewar\OneDrive\Desktop\shAI Club\emnist images\H.PNG")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# resized = cv2.resize(gray, (28,28))
# print(cv2.imwrite(r"C:\Users\sewar\OneDrive\Desktop\shAI Club\emnist images\H2.PNG", resized))


# -------------------------------
predictor = (model.predict(X_test) > 0.5).astype("int32")
matrix = confusion_matrix(y_test.argmax(axis=1), predictor.argmax(axis=1))
print(matrix)
print("----------------------------------")

print(classification_report(y_test.argmax(axis=1), predictor.argmax(axis=1)))
