from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Input, Flatten, Dense, Dropout, Conv2D
from tensorflow.keras.optimizers import Adam

def comparison_siamese_model(input_shape):
    """Create a siamese model for image comparison using VGG19 as base model.
    
    Args:
        input_shape (tuple): Shape of the input images.
    
    Returns:
        keras.models.Model: The compiled siamese model.
    """
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
