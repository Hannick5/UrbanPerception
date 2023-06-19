import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def data_aug(image1_array, image2_array, labels, save_folder): 
    # Définir les paramètres de data augmentation pour chaque image
    datagen1 = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2
    )

    datagen2 = ImageDataGenerator(
        rotation_range=10,
        horizontal_flip=True,
        zoom_range=0.1
    )

    # Convertir les listes d'images et de labels en tableaux numpy
    image1_array = np.array(image1_array)
    image2_array = np.array(image2_array)
    labels = np.array(labels)

    # Obtenir le nombre total d'images
    num_images = len(image1_array)

    # Générer les nouvelles images augmentées
    gen_image1 = datagen1.flow(image1_array, labels, shuffle=False, batch_size=num_images)
    gen_image2 = datagen2.flow(image2_array, labels, shuffle=False, batch_size=num_images)

    # Obtenir les nouvelles images augmentées
    new_image1_array, new_labels = next(gen_image1)
    new_image2_array, _ = next(gen_image2)  # Ignorer les labels générés par datagen2

    # Concaténer les nouvelles images générées avec les anciennes images d'origine
    new_image1_array = np.concatenate((image1_array, new_image1_array), axis=0)
    new_image2_array = np.concatenate((image2_array, new_image2_array), axis=0)

    # Concaténer les labels d'origine avec les nouveaux labels générés
    new_labels = np.concatenate((labels, new_labels), axis=0)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Save each augmented image in npy format
    for i, image in enumerate(new_image1_array):
        np.save(os.path.join(save_folder, f'image1_{i}.npy'), image)

    for i, image in enumerate(new_image2_array):
        np.save(os.path.join(save_folder, f'image2_{i}.npy'), image)
    
    np.save(os.path.join(save_folder, 'new_labels.npy'), new_labels)
    
    return new_image1_array, new_image2_array, new_labels

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

    
#     image1_array = tf.convert_to_tensor(image1_array, dtype=tf.float32)
#     image2_array = tf.convert_to_tensor(image2_array, dtype=tf.float32)
#     labels_formatted = tf.convert_to_tensor(labels_formatted, dtype=tf.float32)
    
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

def load_data(folder):
    new_image1_array = []
    new_image2_array = []
    new_labels = np.load(os.path.join(folder, 'new_labels.npy'), allow_pickle=True)
    
    # Load image1_array
    for i in range(len(new_labels)):
        image_path = os.path.join(folder, f'image1_{i}.npy')
        image = np.load(image_path, allow_pickle=True).astype(np.float32)
        new_image1_array.append(image)
    new_image1_array = np.array(new_image1_array)

    # Load image2_array
    for i in range(len(new_labels)):
        image_path = os.path.join(folder, f'image2_{i}.npy')
        image = np.load(image_path, allow_pickle=True).astype(np.float32)
        new_image2_array.append(image)
    new_image2_array = np.array(new_image2_array)

    return new_image1_array, new_image2_array, new_labels