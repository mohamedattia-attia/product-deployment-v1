from flask import Flask, request, jsonify, render_template
from data_preparation import create_siamese_network
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import os
import uuid  # Module for generating unique IDs
import requests

# Define input shape for the images
input_shape = (224, 224, 3)

# Create the Siamese network
siamese_model = create_siamese_network(input_shape)

# Compile the model
siamese_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

app = Flask(__name__)

def process_uploaded_image(image_id, product_mapping):
    """Process the uploaded image to find similar images."""
    similar_product_info = []

    img_path = f"uploads/{image_id}.jpg"
    img = image.load_img(img_path, target_size=input_shape[:2])
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    embedding = siamese_model.predict([img, img])

    for product_image, product_id in product_mapping.items():
        product_embedding = get_embedding(product_image)  # Get embedding of product image
        similarity = cosine_similarity(embedding, product_embedding)
        if similarity > 0.8:  # Adjust similarity threshold as needed
            similar_product_info.append({"product_image": product_image, "product_id": product_id})

    return similar_product_info

def get_embedding(image_path):
    """Get the embedding of an image."""
    img = image.load_img(image_path, target_size=input_shape[:2])
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return siamese_model.predict([img, img])

def fetch_product_mapping():
    """Fetch product image mappings dynamically from the backend."""
    backend_api_url = 'https://al7rm.com/admin-panel/public/api/product_mapping'
    try:
        response = requests.get(backend_api_url)
        data = response.json()
        # Process the received data as needed
        # For example, extract image filenames and their corresponding IDs
        return data.get('product_mapping', {})
    except Exception as e:
        print(f"Error fetching product mapping: {e}")
        return {}

@app.route('/')
def home():
    """Render the index.html template."""
    return render_template("index.html")

@app.route('/compare_images', methods=['POST'])
def compare_images():
    """Compare the uploaded image with product images."""
    uploaded_image = request.files['image']
    
    # Generate a unique ID for the uploaded image
    image_id = str(uuid.uuid4())
    # Save the uploaded image with the generated ID as the filename
    uploaded_image.save(os.path.join("uploads", f"{image_id}.jpg"))
    
    product_mapping = fetch_product_mapping()
    similar_product_info = process_uploaded_image(image_id, product_mapping)
    
    return jsonify(similar_product_info)

@app.route('/api/compare_images', methods=['POST'])
def compare_images_api():
    """API endpoint for comparing images."""
    data = request.get_json(force=True)
    uploaded_image_path = data.get('image_path')
    uploaded_image_embedding = get_embedding(uploaded_image_path)
    product_mapping = fetch_product_mapping()
    
    similar_product_info = []
    for product_image, product_id in product_mapping.items():
        product_embedding = get_embedding(product_image)
        similarity = cosine_similarity(uploaded_image_embedding, product_embedding)
        if similarity > 0.8:  # Adjust similarity threshold as needed
            similar_product_info.append({"product_image": product_image, "product_id": product_id})

    return jsonify(similar_product_info)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
