import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

def create_siamese_network(input_shape):
    # Base convolutional neural network (CNN)
    base_network = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu')
    ])
    
    # Define inputs for the Siamese network
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    
    # Process inputs through the base network
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    # Calculate Euclidean distance between processed inputs
    distance = tf.keras.layers.Lambda(lambda x: tf.square(x[0] - x[1]))([processed_a, processed_b])
    
    # Create Siamese network model
    siamese_network = models.Model(inputs=[input_a, input_b], outputs=distance)
    
    return siamese_network

# Define input shape for the images
input_shape = (224, 224, 3)

# Create the Siamese network
siamese_model = create_siamese_network(input_shape)

# Compile the model
siamese_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
siamese_model.summary()
