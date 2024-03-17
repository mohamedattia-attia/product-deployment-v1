from flask import Flask, request, jsonify
from data_preparation import create_siamese_network
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Placeholder dictionary to map product image filenames to their IDs
product_mapping = {
    "product_image_1.jpg": "product_id_1",
    "product_image_2.jpg": "product_id_2",
    # Add more mappings as needed
}

# Define input shape for the images
input_shape = (224, 224, 3)

# Create the Siamese network
siamese_model = create_siamese_network(input_shape)

# Compile the model
siamese_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

app = Flask(__name__)

# Function to process uploaded image and find similar images
def process_uploaded_image(uploaded_image):
    # Placeholder list to store similar product image filenames and IDs
    similar_product_info = []

    # Load and preprocess the uploaded image
    img = image.load_img(uploaded_image, target_size=input_shape[:2])
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values to [0, 1]

    # Compute embeddings for the uploaded image
    embedding = siamese_model.predict([img, img])  # Use the same image as both inputs for inference

    # Placeholder logic for similarity comparison
    # Here you would implement the logic to compare embeddings and find similar images
    # For now, we'll just return the product images and their IDs as placeholders
    for product_image, product_id in product_mapping.items():
        # Placeholder logic to compare embeddings (e.g., using cosine similarity)
        # For now, we'll assume all embeddings are the same
        if np.array_equal(embedding, product_embedding):
            similar_product_info.append({"product_image": product_image, "product_id": product_id})

    return similar_product_info

@app.route('/', methods=['POST'])
def compare_images():
    # Get the uploaded image from the request
    uploaded_image = request.files['uploaded_image']

    # Process the uploaded image to find similar images
    similar_product_info = process_uploaded_image(uploaded_image)

    # Return the similar images and their IDs as a JSON response
    return jsonify(similar_product_info)

if __name__ == '__main__':
    app.run(debug=True)
