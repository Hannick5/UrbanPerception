import os
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import glob 
import shutil
import datetime

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Subtract, concatenate, Input, Flatten, Activation, Dense, Dropout, Lambda, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping



def comparison_siamese_model(input_shape):

    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers[:-4]:
        layer.trainable=False

    # Create inputs for pairs of images
    input_1 = Input(shape=input_shape)
    input_2 = Input(shape=input_shape)

    # Get embeddings of the images using the shared VGG19 model
    output_1 = base_model(input_1)
    output_2 = base_model(input_2)

    concat = concatenate([output_1, output_2])

    # Classification layer to predict similarity
    flatten = Flatten()(concat)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(concat)
    x = Dropout(0.3)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    output = Dense(2, activation='sigmoid')(x)

    # Create the complete siamese model
    siamese_model = Model(inputs=[input_1, input_2], outputs=output)
    # Compile the model
    siamese_model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.000001), metrics=['accuracy'])

    # Print model summary
    siamese_model.summary()
    
    return siamese_model
