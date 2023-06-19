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

def prepare_label_for_ranking(labels):
    # Format the labels of left and right images
    labels_formatted = []

    for label in labels:
        if label == "left":
            labels_formatted.append(0)  # 0 represents left image
        elif label == "right":
            labels_formatted.append(1)  # 1 represents right image

    labels_formatted = np.array(labels_formatted)
    labels_formatted = tf.convert_to_tensor(labels_formatted)
    # Split the data into training, validation, and test sets using array slicing
    train_size = int(0.6 * len(labels))
    valid_size = int(0.2 * len(labels))
    
    y_train = labels_formatted[:train_size]
    y_valid = labels_formatted[train_size:train_size + valid_size]
    y_test = labels_formatted[train_size + valid_size:]
    
    return y_train, y_valid, y_test

def create_ranking_network(img_size):
    """
    Create ranking network which give a score to an image.

    :param img_size: size of input images during training
    :type img_size: tuple(int)
    :return: ranking network model
    :rtype: keras.Model
    """
    # Create feature extractor from VGG19
    feature_extractor = VGG19(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    for layer in feature_extractor.layers[:-4]:
        layer.trainable = False

    # Add dense layers on top of the feature extractor
    inp = Input(shape=(img_size, img_size, 3), name='input_image')
    base = feature_extractor(inp)
    base = Flatten(name='Flatten')(base)

    # Block 1
    base = Dense(32, activation='relu', name='Dense_1')(base)
    base = BatchNormalization(name='BN1')(base)
    base = Dropout(0.490, name='Drop_1')(base)

    # Block 2
    base = Dense(128, activation='relu', name='Dense_2')(base)
    base = BatchNormalization(name='BN2')(base)
    base = Dropout(0.368, name='Drop_2')(base)

    # Final dense
    base = Dense(1, name="Dense_Output")(base)
    base_network = Model(inp, base, name='Scoring_model')
    return base_network


def create_meta_network(img_size, weights=None):
    """
    Create meta network which is used to to teach the ranking network.

    :param img_size: dimension of input images during training.
    :type img_size: tuple(int)
    :param weights: path to the weights use for initialization
    :type weights: str
    :return: meta network model
    :rtype: keras.Model
    """

    # Create the two input branches
    input_left = Input(shape=(img_size, img_size, 3), name='left_input')
    input_right = Input(shape=(img_size, img_size, 3), name='right_input')
    base_network = create_ranking_network(img_size)
    left_score = base_network(input_left)
    right_score = base_network(input_right)

    # Subtract scores
    diff = Subtract()([left_score, right_score])

    # Pass difference through sigmoid function.
    prob = Activation("sigmoid", name="Activation_sigmoid")(diff)
    model = Model(inputs=[input_left, input_right], outputs= prob, name="Meta_Model")

    if weights:
        print('Loading weights ...')
        model.load_weights(weights)


    sgd = SGD(learning_rate=1e-6, decay=1e-6, momentum=0.393, nesterov=True)
    model.compile(optimizer=Adam(learning_rate=0.000001), loss="binary_crossentropy", metrics=['accuracy'])

    return model