# ----------------------------------------#
#           Importing libraries           #
# ----------------------------------------#

import os
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import Adam

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# ----------------------------------------#
#           Importing data                #
# ----------------------------------------#

print("Importing data...")

data = pd.read_csv("data\question_1\duels_question_1.csv",usecols=[0,1,2], header=None)
data.columns = ["Image 1", "Image 2", "labels"]

#Deleting the no preference data
data = data[data["labels"] != "No preference"]

# ----------------------------------------#
#        Preprocessing the data           #
# ----------------------------------------#

print("Processing datasets...")

def normalized(rgb):
    # Variable initialization
    norm = np.zeros((rgb.shape[0], rgb.shape[1], 3), np.float32)

    # Get channel order for cv2
    b = rgb[:, :, 0]
    g = rgb[:, :, 1]
    r = rgb[:, :, 2]

    # Normalize values thanks to histogram equalization
    norm[:, :, 0] = cv2.equalizeHist(b)
    norm[:, :, 1] = cv2.equalizeHist(g)
    norm[:, :, 2] = cv2.equalizeHist(r)
    return norm

# Specify the directory where your images are stored
image_folder = "data\question_1\Sample_web_green"

image1_names = data.iloc[:,0].values
image2_names = data.iloc[:,1].values
labels = data.iloc[:,2].values

image1_array = []
image2_array = []

for image1_name, image2_name in zip(image1_names, image2_names):
    for filename in os.listdir(image_folder):
        if image1_name in filename:
            image1_path = os.path.join(image_folder, filename)
            image1 = cv2.imread(image1_path)
            image1 = cv2.resize(image1, (300, 300))
            image1 = normalized(image1)
            image1_array.append(image1)
        elif image2_name in filename:
            image2_path = os.path.join(image_folder, filename)
            image2 = cv2.imread(image2_path)
            image2 = cv2.resize(image2, (300, 300))
            image2 = normalized(image2)
            image2_array.append(image2)

# Format the labels of left and right images
labels_formatted = []

for label in labels:
    if label == "left":
        labels_formatted.append([1,0])  # 0 represents left image
    elif label == "right":
        labels_formatted.append([0,1])  # 1 represents right image

labels_formatted = np.array(labels_formatted)

# Conversion of the lists into numpy arrays
image1_array = np.array(image1_array)
image2_array = np.array(image2_array)

# Split the data into training, validation, and test sets using array slicing
train_size = int(0.6 * len(image1_array))
valid_size = int(0.2 * len(image1_array))

X_train = [image1_array[:train_size], image2_array[:train_size]]
y_train = labels_formatted[:train_size]

X_valid = [image1_array[train_size:train_size + valid_size], image2_array[train_size:train_size + valid_size]]
y_valid = labels_formatted[train_size:train_size + valid_size]

X_test = [image1_array[train_size + valid_size:], image2_array[train_size + valid_size:]]
y_test = labels_formatted[train_size + valid_size:]

# ----------------------------------------#
#        Building the siamese model       #
# ----------------------------------------#

# Build the siamese network model with VGG19
input_shape = image1_array[0].shape  # Get the size of a preprocessed image
base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

for layer in base_model.layers:
    layer.trainable=False

# Create inputs for pairs of images
input_1 = Input(shape=input_shape)
input_2 = Input(shape=input_shape)

# Get embeddings of the images using the shared VGG19 model
output_1 = base_model(input_1)
output_2 = base_model(input_2)

concat = concatenate([output_1,output_2])

# Classification layer to predict similarity
flatten = Flatten()(concat)
dense1 = Dense(128, activation='relu')(flatten)
dropout = Dropout(0.5)(dense1)
output = Dense(1, activation='sigmoid')(dropout)

# Create the complete siamese model
siamese_model = Model(inputs=[input_1, input_2], outputs=output)

# Compile the model
siamese_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Print model summary
siamese_model.summary()

print("Training model")

# Train the siamese network
siamese_model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_valid, y_valid))