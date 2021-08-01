# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import p_config as config
import numpy as np
import pickle
import os
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
import random

FLOWERS_ORIG_DIR = config.FLOWERS_ORIG_DIR
EPOCHS=config.EPOCHS
     

def csv_feature_generator(inputPath, bs, numClasses, mode="train"):
     # open the input file for reading
     f = open(inputPath, "r")

     # loop indefinitely
     while True:
          # init our batch of data and labels
          data = []
          labels = []

          # keep looping until we reach our batch size
          while len(data) < bs:
               # attempt to read the next row of the CSV file
               row = f.readline()

               # check to see if the row is empty, indicating we have
               # reached the end of the file
               if row == "":
                    # reset the file pointer to the beginning of the file
                    # and re-read the row
                    f.seek(0)
                    row = f.readline()

                    # if we are evaluating we should now break from our
                    # loop to ensure we don't continue to fill up the
                    # batch from samples at the beginning of the file
                    if mode == "eval":
                         break

               # extract the class label and features from the row
               row = row.strip().split(",")
               label = row[1]
               
               # one hot encode the output variable
               # this ensures that each example has an expected probability
               # of 1.0 for the actual class value and an expected probability
               # of 0.0 for all other class values, achieved using the
               # to_categorical()
               # Keras function
               label = to_categorical(label, num_classes=numClasses)
               features = np.array(row[2:], dtype="float")
               
               # update the data and label lists
               data.append(features)
               labels.append(label)
          
          # yield the batch to the calling function
          yield (np.array(data), np.array(labels))


# load the label encoder from disk
le = pickle.loads(open(FLOWERS_ORIG_DIR / config.LE_PATH, "rb").read())

# derive the paths to the training and testing CSV files
trainPath = os.path.sep.join([str(FLOWERS_ORIG_DIR / config.BASE_CSV_PATH),
     "{}.csv".format(config.TRAIN)])
valPath = os.path.sep.join([str(FLOWERS_ORIG_DIR / config.BASE_CSV_PATH), 
     "{}.csv".format(config.VAL)])
testPath = os.path.sep.join([str(FLOWERS_ORIG_DIR / config.BASE_CSV_PATH),
     "{}.csv".format(config.TEST)])

# determine the total number of images in the training and validation sets
totalTrain = sum([1 for l in open(trainPath)])
totalVal = sum([1 for l in open(valPath)])

# extract the testing labels from the CSV file and then determine the
# number of testing images
testLabels = [int(row.split(",")[1]) for row in open(testPath)]
testImagePaths = [row.split(',')[0] for row in open(testPath)]
totalTest = len(testLabels)
print(f'Total testing images: {totalTest}')

# construct the training, validation, and testing generators
trainGen = csv_feature_generator(trainPath, config.BATCH_SIZE,
     len(config.CLASSES), mode="train")
valGen = csv_feature_generator(valPath, config.BATCH_SIZE,
     len(config.CLASSES), mode="eval")
testGen = csv_feature_generator(testPath, config.BATCH_SIZE,
     len(config.CLASSES), mode="eval")

# the sequential class groups a linear stack of layers into a tf.keras.Model
model = Sequential()

# The add method adds a layer instance on top of the layer stack
# `Dense` represents a densely-connected NN layer

# Adding the kwarg `input_shape` creates an input layer to insert before the
# current layer, which is equivalent to explicitly defining an `InputLayer`.

# The first argument is the units, a positive integer that represents the
# dimensionality of the output space.  Also, this represents the number of neurons in the layer.

# The shape to the input to the model is defined as an argument to the first hidden layer.

# The line of code that adds the first Dense layer defines the input or visible
# layer and the first hidden layer.
model.add(Dense(256, input_shape=(7 * 7 * 2048,), activation="relu"))

# This is a hidden layer
model.add(Dense(16, activation="relu"))

# The output layer has as many nodes as the number of classes (5).
# The `categorical_crossentropy` function requires that the output layer is
# configured with the `softmax` activation to predict the probability for each class
model.add(Dense(len(config.CLASSES), activation="softmax"))

# compiling the model
# the model is fit using stochastic gradient descent, lr = learning rate
opt = SGD(lr=1e-3, momentum=0.9, decay=1e-3 / 25)

# Compiling the model uses the efficient numerical libraries under the covers, such as Theano or TensorFlow

# Training a network means finding the best set of weights to map inputs to outputs in our dataset

# We must specify the loss function to evaluate the weights, the optimizer to
# search through different weights for the network, and any optional metrics
# we want to collect and report during training
# if you have greater than 2 classes you should use 'categorical_crossentropy'
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
# the model is fit for the specified epochs on the training data
# the validation data is used as well, allowing us to evaluate both loss
# and classification accuracy on the train and test sets at the end of each
# training epoch and draw learning curves.
print("[INFO] training simple network...")
H = model.fit(x=trainGen,
     steps_per_epoch=totalTrain // config.BATCH_SIZE,
     validation_data=valGen,
     validation_steps=totalVal // config.BATCH_SIZE,
     epochs=EPOCHS)

train_loss, train_acc = model.evaluate(x = trainGen, steps=(totalTrain // config.BATCH_SIZE))

valid_loss, valid_acc = model.evaluate(x = valGen, steps = (totalVal // config.BATCH_SIZE))

print()
print(f'Train Loss: {train_loss}')
print(f'Train accuracy: {train_acc}') 
print()

print(f'Valid Loss: {valid_loss}')
print(f'Valid accuracy: {valid_acc}') 
print()

# make predictions on the testing images, finding the index of the
# label with the corresponding largest predicted probability, then
# show a nicely formatted classification report
print("[INFO] evaluating network...")
predIdxs = model.predict(x=testGen,steps=(totalTest // config.BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)

# classification_report(y_true, y_pred, target_names)
print(classification_report(testLabels, predIdxs, target_names=le.classes_))

# create dataset for print images
image_label_pred_list = list(zip(testImagePaths, testLabels, predIdxs))

# set the random seed
random.seed(config.RANDOM_SEED)
# shuffle the list to vary the output
random.shuffle(image_label_pred_list)
flower_types = config.CLASSES
le = LabelEncoder()
le.fit(flower_types)

# import library
import matplotlib.pyplot as plt

# create figure
plt.figure(figsize=(10,10))

# plot images with titles
for i in range(25):
     ax = plt.subplot(5, 5, i + 1)
     pic = plt.imread(image_label_pred_list[i][0])
     plt.imshow(pic)
     plt.title(f'Label: {le.inverse_transform([image_label_pred_list[i][1]])} \
\nPrediction: {le.inverse_transform([image_label_pred_list[i][2]])}')
     plt.axis('off')
          
# show image
plt.show()

# serialize the model to disk
print("[INFO] saving model...")
f = open(FLOWERS_ORIG_DIR / config.MODEL_PATH, "wb")
f.write(pickle.dumps(model))
f.close()


    




