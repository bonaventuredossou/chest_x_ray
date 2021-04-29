import argparse
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from google.colab import drive
from model import XRAY

EPOCHS = 500
INIT_LR = 1e-3
BS = 64
num_classes = 2
width = 128
height = 128
depth = 1

drive.mount("/content/drive")


data = np.load("/content/drive/My Drive/chest_xray/pneumonia_training.npy")
labels = np.load("/content/drive/My Drive/chest_xray/pneumonia_labels.npy")

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing

# print(labels)
# print(data.shape)
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.2, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=num_classes)
testY = to_categorical(testY, num_classes=num_classes)

# # construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.2,
                         height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=False, fill_mode="nearest", vertical_flip=False)

# initialize the model
print("[INFO] compiling model...")
model_1 = XRAY.build(width=width, height=height, depth=depth, classes=num_classes)

opt = SGD(INIT_LR, 0.9)
model_1.compile(loss="categorical_crossentropy", optimizer=opt,
                metrics=["accuracy"])

# train the network
print("[INFO] training network...")
# saves the model weights after each epoch if the validation loss decreased
#  the entire model will be saved to the file “best_model.h5” only when accuracy on the validation dataset improves
# overall across the entire training process.
checkpointer = ModelCheckpoint(filepath=args["model"], verbose=1, save_best_only=True, monitor='val_accuracy',
                               mode='max')

# As soon as the loss of the model begins to increase on the test dataset, we will stop training.
# First, we can define the early stopping callback.
# Simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500)

# H = model_1.fit(aug.flow(trainX, trainY, batch_size=BS), epochs=EPOCHS, verbose=1,
#                           validation_data=(testX, testY), callbacks=[checkpointer, es],
#                           steps_per_epoch=len(trainX) // BS)

H = model_1.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, verbose=1,
                validation_data=(testX, testY), callbacks=[checkpointer, es])

# save the model to disk
print("[INFO] serializing network...")
model_1.save(args["model"])

# plot the training loss and accuracy
plt.style.use("dark_background")
plt.figure()
N = EPOCHS
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.plot(H.history["acc"], label="train_acc")
plt.plot(H.history["val_acc"], label="val_acc")
plt.title("Training Log_loss/Accuracy of F-SER20")
plt.xlabel("Training Epochs")
plt.ylabel("Log_loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
plt.show()
