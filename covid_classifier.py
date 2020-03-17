#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import keras as keras
from chexnet.models.keras import ModelFactory
import os
from configparser import ConfigParser
from keras.models import Model
from keras.layers import Concatenate
from keras.layers import Activation, Input, Lambda, BatchNormalization, GlobalAveragePooling2D,Dense
from keras.layers import MaxPool2D, UpSampling2D, AveragePooling2D, Flatten
from keras.layers import Conv2D, SeparableConv2D, Dropout, Dense
from keras.layers import MaxPooling2D
from keras.layers import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal, constant
from keras.layers import LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imutils import paths
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
import seaborn as sns


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


config_file = "./sample_config.ini"
cp = ConfigParser()
cp.read(config_file)

# default config
output_dir = cp["DEFAULT"].get("output_dir")
base_model_name = cp["DEFAULT"].get("base_model_name")
class_names = cp["DEFAULT"].get("class_names").split(",")
image_source_dir = cp["DEFAULT"].get("image_source_dir")
image_dimension = cp["TRAIN"].getint("image_dimension")

model_weights_path = os.path.join(os.getcwd(), "base_model.h5")
print(model_weights_path)

base_model_name = "DenseNet121"

model_factory = ModelFactory()
model = model_factory.get_model(
    class_names,
    model_name=base_model_name,
    use_base_weights=False,
    weights_path=model_weights_path)

# construct the head of the model that will be placed on top of the
# the base model
headModel = model.get_layer('relu').output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

new_model = Model(inputs=model.input, outputs=headModel)
#new_model.summary()



# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 25
BS = 8

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images('./dataset'))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 255]
data = np.array(data) / 255.0
labels = np.array(labels)
print(labels)
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


# partition the data into training and testing splits using 70% of
# the data for training and the remaining 30% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.30, stratify=labels,
                                                  random_state=42)

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
    rotation_range=15,
    fill_mode="nearest")

for layer in model.layers:
    layer.trainable = False

opt = keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
new_model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = new_model.fit_generator(
    trainAug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = new_model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax)  # annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['covid', 'normal']); ax.yaxis.set_ticklabels(['covid', 'normal']);
plt.show()


# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training Loss on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("plot_Loss.png")


N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("plot_accuracy.png")


tmodel = Model(inputs=new_model.inputs, outputs=model.get_layer('bn').output)
#tmodel.summary()

image2 = cv2.imread('./dataset/covid/t1.jpeg')
image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image2 = cv2.resize(image2, (224, 224))

im = np.expand_dims(image,axis=0) 
im = tmodel.predict(im)

print(im.shape)
im = im[0, :, :, :]

cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
#weights softmax
cam = np.zeros(dtype=np.float32, shape=(im.shape[:2]))
for i in range(im.shape[2]):
    cam += im[:, :, i]

cam /= np.max(cam)
heatmap = cam
heatmap[np.where(cam < 0.9)] = 0
heatmap = cv2.resize(cam, (224, 224))
heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
img = heatmap * 0.5 + image2


cv2.imshow("frame",img/255)
cv2.waitKey(0)
#serialize the model to disk

print("[INFO] saving COVID-19 detector model...")
new_model.save_weights("covid-19_model.h5")
