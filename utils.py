import matplotlib.pyplot as plt
import numpy as np

def test_accuracy(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    
def plot_loss(history):
    # Historique des valeurs de précision d'entraînement et de validation
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Historique des numéros d'époque
    epochs = range(1, len(train_loss) + 1)

    # Tracer la courbe de précision d'entraînement
    plt.plot(epochs, train_loss, 'b', label='Train Loss')
    # Tracer la courbe de précision de validation
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save loss curve
    plt.savefig('loss_curve.png')

    # Afficher le graphique
    plt.show()
    
    
def plot_accuracy(history):
    # Historique des valeurs de précision d'entraînement et de validation
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # Historique des numéros d'époque
    epochs = range(1, len(train_accuracy) + 1)

    # Tracer la courbe de précision d'entraînement
    plt.plot(epochs, train_accuracy, 'b', label='Train Accuracy')
    # Tracer la courbe de précision de validation
    plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Save accuracy curve
    plt.savefig('accuracy_curve.png')

    # Afficher le graphique
    plt.show()
    
def predicting_on_dataset(X_pred, model):
    # Select a subset of the test data for visualization
    subset_size = 300
    X_subset = X_pred[:subset_size]

    # Make predictions on the subset of test data
    predictions = model.predict(X_subset)

    # Plot the images and predictions
    fig, axes = plt.subplots(subset_size, 2, figsize=(10, subset_size*2))
    for i in range(subset_size):
        # Plot first image
        axes[i, 0].imshow(X_subset[0][i])
        axes[i, 0].axis('off')

        # Plot second image
        axes[i, 1].imshow(X_subset[1][i])
        axes[i, 1].axis('off')

        # Add predicted score as title
        score = predictions[i]  # Assuming the second element represents the score
        axes[i, 1].set_title(score)

    
    plt.tight_layout()
    # Save 
    plt.savefig('predict_result')
    plt.show()
    

def plot_ranking_predict(ranking_model, X_pred, save_path):
    # Predict scores for the images
    scores = ranking_model.predict(X_pred[0])

    # Create an array of indices to maintain the original order
    indices = np.arange(len(scores))

    # Sort the indices based on the scores in descending order
    sorted_indices = sorted(indices, key=lambda x: scores[x], reverse=True)

    # Set the number of columns for the grid
    num_columns = 5

    # Calculate the number of rows based on the number of images and columns
    num_images = len(X_pred[0])
    num_rows = int(np.ceil(num_images / num_columns))

    # Create a figure and axes for the grid
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 3*num_rows))

    # Iterate over the sorted indices and plot the images in the grid
    for i, index in enumerate(sorted_indices):
        row = i // num_columns
        col = i % num_columns

        # Plot the image with the corresponding score
        ax = axes[row, col]
        ax.imshow(X_pred[0][index])
        ax.axis('off')
        ax.set_title(f"Score: {scores[index]}")

    # Adjust the layout and display the grid of images
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()