import pandas as pd
import numpy as np 
import cv2
import os
import glob
import tensorflow as tf
from tqdm import tqdm

def prepare_data(folder_path):
    data = pd.read_csv(folder_path, usecols=[0,1,2], header=None)
    data.columns = ["Image 1", "Image 2", "labels"]

    #Deleting the no preference data
    data = data[data["labels"] != "No preference"]

    return data


def prepare_dataset_arrays(image_folder, data, shape):

    image1_names = data.iloc[:,0].values
    image2_names = data.iloc[:,1].values
    labels = data.iloc[:,2].values

    image1_array = []
    image2_array = []

    total_files = len(image1_names) + len(image2_names)
    progress = tqdm(total=total_files, desc="Processing Images", unit="image")
    
    for image1_name, image2_name in zip(image1_names, image2_names):
        for filename in os.listdir(image_folder):
            if image1_name in filename:
                image1_path = os.path.join(image_folder, filename)
                image1 = cv2.imread(image1_path)
                image1 = cv2.resize(image1, (shape, shape))
                image1 = image1.astype(np.float32) / 255.0
                image1_array.append(image1)
                progress.update(1)
            elif image2_name in filename:
                image2_path = os.path.join(image_folder, filename)
                image2 = cv2.imread(image2_path)
                image2 = cv2.resize(image2, (shape, shape))
                image2 = image2.astype(np.float32) / 255.0
                image2_array.append(image2)
                progress.update(1)
        
    progress.close()
                
    return image1_array, image2_array, labels

def prepare_prediction_siamese(directory, shape):
    image_pred = []
    total_files = len(glob.glob(directory))
    progress = tqdm(total=total_files, desc="Processing Images", unit="image")

    for img in glob.glob(directory):
        image1 = cv2.imread(img)
        image1 = cv2.resize(image1, (shape, shape))
        image1 = image1.astype(np.float32) / 255.0
        image_pred.append(image1)
        progress.update(1)

    progress.close()

    image_pred_1 = tf.convert_to_tensor(np.array(image_pred[:300]))
    image_pred_2 = tf.convert_to_tensor(np.array(image_pred[300:600]))

    X_pred = [image_pred_1, image_pred_2]

    return X_pred

def prepare_prediction_siamese_mapillary(directory, shape):
    image_pred = []
    files = os.listdir(directory)
    total_files = len(files)
    progress = tqdm(total=total_files, desc="Processing Images", unit="image")

    for filename in files:
        img_file = os.path.join(directory, filename)
        image1 = cv2.imread(img_file)
        image1 = cv2.resize(image1, (shape, shape))
        image1 = image1.astype(np.float32) / 255.0
        image_pred.append(image1)
        progress.update(1)

    progress.close()

    X_pred = tf.convert_to_tensor(np.array(image_pred))

    return X_pred



def prepare_dataset_for_network(image1_array, image2_array, labels):
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

#     labels_formatted = tf.convert_to_tensor(labels_formatted)

#     image1_array = tf.convert_to_tensor(image1_array)
#     image2_array = tf.convert_to_tensor(image2_array)
    
    # Split the data into training, validation, and test sets using array slicing
    train_size = int(0.6 * len(image1_array))
    valid_size = int(0.2 * len(image1_array))

    X_train = [image1_array[:train_size], image2_array[:train_size]]
    y_train = labels_formatted[:train_size]

    X_valid = [image1_array[train_size:train_size + valid_size], image2_array[train_size:train_size + valid_size]]
    y_valid = labels_formatted[train_size:train_size + valid_size]

    X_test = [image1_array[train_size + valid_size:], image2_array[train_size + valid_size:]]
    y_test = labels_formatted[train_size + valid_size:]
    
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

def prepare_dataset_generators(image1_array, image2_array, labels, batch_size, model_type):
    labels_formatted = []
    
    if model_type == "comparison":

        for label in labels:
            if label == "left":
                labels_formatted.append([1, 0])
            elif label == "right":
                labels_formatted.append([0, 1])
    elif model_type == "ranking":
         for label in labels:
            if label == "left":
                labels_formatted.append(1)
            elif label == "right":
                labels_formatted.append(0)

    labels_formatted = np.array(labels_formatted)
    image1_array = np.array(image1_array)
    image2_array = np.array(image2_array)

    train_size = int(0.6 * len(image1_array))
    valid_size = int(0.2 * len(image1_array))

    def data_generator(images1, images2, labels):
        num_samples = len(labels)
        while True:
            indices = np.random.permutation(num_samples)
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_images1 = images1[batch_indices]
                batch_images2 = images2[batch_indices]
                batch_labels = labels[batch_indices]
                yield [batch_images1, batch_images2], batch_labels

    train_generator = data_generator(image1_array[:train_size], image2_array[:train_size], labels_formatted[:train_size])
    valid_generator = data_generator(image1_array[train_size:train_size + valid_size], image2_array[train_size:train_size + valid_size], labels_formatted[train_size:train_size + valid_size])
    test_generator = data_generator(image1_array[train_size + valid_size:], image2_array[train_size + valid_size:], labels_formatted[train_size + valid_size:])

    return train_generator, valid_generator, test_generator, train_size, valid_size