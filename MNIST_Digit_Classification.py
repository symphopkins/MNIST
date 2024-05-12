# -*- coding: utf-8 -*-

#importing libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

#loading the MNIST digits classification dataset from Tensorflow Keras
mnist = tf.keras.datasets.mnist

#splitting data into training and test datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#we are dividing the values by 255.0 to normalize the pixels so that they are within the range of 0 and 1, and to convert the values to floats
x_train, x_test = x_train / 255.0, x_test / 255.0

#checking shape of data
print(f'''
x_train shape: {x_train.shape}
y_train shape: {y_train.shape}
x_test shape:  {x_test.shape}
y_test shape:  {y_test.shape}
''')

#checking data type of each variable
print(f'''
x_train data type: {x_train.dtype}
y_train data type: {y_train.dtype}
x_test data type:  {x_test.dtype}
y_test data type:  {y_test.dtype}
''')

# importing libraries
import numpy as np
import matplotlib.pyplot as plt

# displaying distribution
num, counts = np.unique(y_train, return_counts=True)

for value, count in zip(num, counts):
    print(f"{value}: {count}")

# plotting distribution
plt.bar(num, counts)
plt.xlabel('Value')
plt.ylabel('Count')
plt.xticks(num)
plt.title('Distribution of Values')
plt.show()


#reshaping the data
#each image is 28x28, so we will set the height and width to be those values
height, width = x_train.shape[1:]
x_train = x_train.reshape(x_train.shape[0], height, width, 1)
x_test = x_test.reshape(x_test.shape[0], height, width, 1)

#checking new shape of data
print(f'''
x_train shape: {x_train.shape}
x_test shape:  {x_test.shape}
''')


#checking pixel value range
import numpy as np
print(f'The minimum value of pixels = {np.amin(x_train[0])}; the maximum value of pixels ={np.amax(x_train[0])}' )


#visualizing an image
import matplotlib.pyplot as plt
plt.imshow(x_train[0], cmap='gray', interpolation='none')
plt.title(f'Digit: {y_train[0]}')
plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)


#creating first model
model = models.Sequential()

#adding first convolution layer
#since we reshaped the x_train set to be (60000,28,28,1), we can simply set the input_shape to be x_train[1:] so that it'll be (28,28,1)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape= x_train.shape[1:]))

#adding the first max pooling layer; pooling window will be 2x2
model.add(layers.MaxPooling2D((2, 2)))

#adding second convolution layer; increasing # of filters from 32 to 64
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#adding second max pooling layer
model.add(layers.MaxPooling2D((2, 2)))

#adding third convolution layer; increasing # of filters from 64 to 128
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

#adding third max pooling layer
model.add(layers.MaxPooling2D((2, 2)))

#we need to change the dimensions of the output array to 2D to make it work with the classification algorithm, so we will add a flatten layer
model.add(layers.Flatten())

#adding layers for classification; layer will contain 128 neurons
model.add(layers.Dense(128, activation='relu'))

#adding dropout layer
model.add(layers.Dropout(0.3))

#since we have 10 labels, we need to have 10 neurons for the final output later; activation=None
model.add(layers.Dense(10))

#configuring model; since activation=None, from_logits must =True
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


model.summary()

# Commented out IPython magic to ensure Python compatibility.
# %%time
# #training the model and storing the history in a variable to plot later
# #we will make the number of epochs short since training models for image classification can run for a long time
# history = model.fit(x_train, y_train, epochs=20, verbose = 1)

#importing libraries
import seaborn as sns
import pandas as pd

#visualizing training history
train_history = pd.DataFrame(history.history)
train_history['epoch'] = history.epoch
#plotting training loss line plot
sns.lineplot(x='epoch', y ='loss', data =train_history)
#displaying line plot
plt.show()

#plotting training accuracy
sns.lineplot(x='epoch', y ='accuracy', data =train_history)
#adding legends
plt.legend(labels=['train_accuracy'])
#displaying line plot
plt.show()

#calculating test error and accuracy
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'''
Test Loss: {round(test_loss, 4)}
Test Accuracy: {round(test_acc, 4)}''')

from sklearn.metrics import classification_report
# creating the classification report
y_pred = model.predict(x_test)
y_pred = tf.nn.softmax(y_pred)
y_pred = np.argmax(y_pred, axis=1)
report = classification_report(y_test, y_pred)

# printing the classification report
print(report)

# importing library
from sklearn.metrics import confusion_matrix

label_names = np.unique(y_test)

# creating confusion matrix
cm = confusion_matrix(y_test, y_pred, normalize='true')

# plotting normalized confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Values')
plt.ylabel('True Values')
plt.title('Normalized Confusion Matrix')
plt.show()

#creating second model
model = models.Sequential()

#adding first convolution layer
#since we reshaped the x_train set to be (60000,28,28,1), we can simply set the input_shape to be x_train[1:] so that it'll be (28,28,1)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape= x_train.shape[1:]))

#adding the first max pooling layer; pooling window will be 2x2
model.add(layers.MaxPooling2D((2, 2)))

#adding second convolution layer; increasing # of filters from 32 to 64
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#adding second max pooling layer
model.add(layers.MaxPooling2D((2, 2)))

#adding third convolution layer; increasing # of filters from 64 to 128
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

#adding third max pooling layer
model.add(layers.MaxPooling2D((2, 2)))

#we need to change the dimensions of the output array to 2D to make it work with the classification algorithm, so we will add a flatten layer
model.add(layers.Flatten())

#adding first layer for classification; layer will contain 128 neurons
model.add(layers.Dense(128, activation='relu'))

#adding first dropout layer
model.add(layers.Dropout(0.3))

#adding second layer for classification; layer will contain 64 neurons
model.add(layers.Dense(64, activation='relu'))

#adding second dropout layer
model.add(layers.Dropout(0.3))

#adding third layer for classification; layer will contain 64 neurons
model.add(layers.Dense(32, activation='relu'))

#adding third dropout layer
model.add(layers.Dropout(0.3))

#since we have 10 labels, we need to have 10 neurons for the final output later; activation=None
model.add(layers.Dense(10))

#configuring model; since activation=None, from_logits must =True
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


model.summary()


# Commented out IPython magic to ensure Python compatibility.
# %%time
# #adding early stopping; if the validation accuracy does not improve for 3 epochs, we will stop training
# callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience= 3)
# 
# #training the model and storing the history in a variable to plot later
# #we will make the number of epochs short since training models for image classification can run for a long time
# history = model.fit(x_train, y_train, epochs=20, callbacks=[callback], verbose = 1)

#visualizing training history
train_history = pd.DataFrame(history.history)
train_history['epoch'] = history.epoch
#plotting training loss line plot
sns.lineplot(x='epoch', y ='loss', data =train_history)
#adding legends
plt.legend(labels=['train_loss'])
#displaying line plot
plt.show()

#plotting training accuracy
sns.lineplot(x='epoch', y ='accuracy', data =train_history)
#adding legends
plt.legend(labels=['train_accuracy', 'val_accuracy'])
#displaying line plot
plt.show()


#calculating test error and accuracy
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'''
Test Loss: {round(test_loss, 4)}
Test Accuracy: {round(test_acc, 4)}''')

# creating the classification report
y_pred = model.predict(x_test)
y_pred = tf.nn.softmax(y_pred)
y_pred = np.argmax(y_pred, axis=1)
report = classification_report(y_test, y_pred)

# printing the classification report
print(report)

# creating confusion matrix
cm = confusion_matrix(y_test, y_pred, normalize='true')

# plotting normalized confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Values')
plt.ylabel('True Values')
plt.title('Normalized Confusion Matrix')
plt.show()

