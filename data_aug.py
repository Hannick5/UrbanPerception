import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def data_aug(image1_array, image2_array, labels, save_folder):
    """Apply data augmentation techniques to images and labels.
    
    Args:
        image1_array (list): List of original image arrays for the first set of images.
        image2_array (list): List of original image arrays for the second set of images.
        labels (list): List of labels corresponding to the images.
        save_folder (str): Path to the folder where augmented images and labels will be saved.
    
    Returns:
        tuple: A tuple containing the new augmented image arrays for image1, image2, and the new labels.
    """
    # Define data augmentation parameters for each image
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

    # Convert image and label lists to numpy arrays
    image1_array = np.array(image1_array)
    image2_array = np.array(image2_array)
    labels = np.array(labels)

    # Get the total number of images
    num_images = len(image1_array)

    # Generate new augmented images
    gen_image1 = datagen1.flow(image1_array, labels, shuffle=False, batch_size=num_images)
    gen_image2 = datagen2.flow(image2_array, labels, shuffle=False, batch_size=num_images)

    # Get the new augmented images
    new_image1_array, new_labels = next(gen_image1)
    new_image2_array, _ = next(gen_image2)  # Ignore labels generated by datagen2

    # Concatenate the newly generated images with the original images
    new_image1_array = np.concatenate((image1_array, new_image1_array), axis=0)
    new_image2_array = np.concatenate((image2_array, new_image2_array), axis=0)

    # Concatenate the original labels with the newly generated labels
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
    """Prepare the dataset for a neural network by formatting labels and splitting the data.
    
    Args:
        image1_array (list): List of image arrays for the first set of images.
        image2_array (list): List of image arrays for the second set of images.
        labels (list): List of labels corresponding to the images.
    
    Returns:
        tuple: A tuple containing the training, validation, and test sets in the form of (X_train, y_train), (X_valid, y_valid), (X_test, y_test).
    """
    # Format the labels of left and right images
    labels_formatted = []

    for label in labels:
        if label == "left":
            labels_formatted.append([1, 0])  # 0 represents left image
        elif label == "right":
            labels_formatted.append([0, 1])  # 1 represents right image

    labels_formatted = np.array(labels_formatted)

    # Convert the lists into numpy arrays
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
    
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def load_data(folder):
    """Load augmented images and labels from a specified folder.
    
    Args:
        folder (str): Path to the folder containing the augmented images and labels.
    
    Returns:
        tuple: A tuple containing the loaded image arrays for image1, image2, and the labels.
    """
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

def apply_random_contrast(image_array):
    """Apply random contrast adjustment to an image array.
    
    Args:
        image_array (numpy.ndarray): Input image array.
    
    Returns:
        numpy.ndarray: Image array with random contrast adjustment.
    """
    # Generate a random contrast value between 0.8 and 1.2
    contrast_factor = np.random.uniform(0.8, 1.2)
    
    # Apply contrast adjustment
    adjusted_image = image_array * contrast_factor
    
    # Clip the pixel values to the valid range [0, 255]
    adjusted_image = np.clip(adjusted_image, 0, 255)
    
    return adjusted_image

def data_aug_with_contrast(image1_array, image2_array, labels, save_folder):
    """Apply data augmentation techniques to images and labels.
    
    Args:
        image1_array (list): List of original image arrays for the first set of images.
        image2_array (list): List of original image arrays for the second set of images.
        labels (list): List of labels corresponding to the images.
        save_folder (str): Path to the folder where augmented images and labels will be saved.
    
    Returns:
        tuple: A tuple containing the new augmented image arrays for image1, image2, and the new labels.
    """
    # Convert image and label lists to numpy arrays
    image1_array = np.array(image1_array)
    image2_array = np.array(image2_array)
    labels = np.array(labels)

    # Apply random contrast adjustment to the image arrays
    new_image1_array = np.array([apply_random_contrast(image) for image in image1_array])
    new_image2_array = np.array([apply_random_contrast(image) for image in image2_array])

    # Concatenate the newly generated images with the original images
    new_image1_array = np.concatenate((image1_array, new_image1_array), axis=0)
    new_image2_array = np.concatenate((image2_array, new_image2_array), axis=0)

    # Concatenate the original labels with the newly generated labels
    new_labels = np.concatenate((labels, labels), axis=0)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Save each augmented image in npy format
    for i, image in enumerate(new_image1_array):
        np.save(os.path.join(save_folder, f'image1_{i}.npy'), image)

    for i, image in enumerate(new_image2_array):
        np.save(os.path.join(save_folder, f'image2_{i}.npy'), image)
    
    np.save(os.path.join(save_folder, 'new_labels.npy'), new_labels)
    
    return new_image1_array, new_image2_array, new_labels
