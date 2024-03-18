from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import List, Dict
from data_preparation import create_siamese_network
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image  # Import Image module from PIL
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import os
import uuid  # Module for generating unique IDs
import json

# Define input shape for the images
input_shape = (224, 224, 3)

# Create the Siamese network
siamese_model = create_siamese_network(input_shape)

# Compile the model
siamese_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

app = FastAPI()

# Define the templates directory
templates = Jinja2Templates(directory="templates")

product_mapping = {}  # Global variable to store the product mapping

def load_product_mapping():
    """Load product mapping from data.json."""
    global product_mapping
    try:
        with open("data.json", "r") as file:
            product_mapping = json.load(file)
    except Exception as e:
        print(f"Error loading product mapping: {e}")

load_product_mapping()

def process_uploaded_image(image_id: str, uploaded_image_path: str) -> List[Dict[str, str]]:
    """Process the uploaded image to find similar images."""
    similar_product_info = []

    # Preprocess uploaded image
    img = Image.open(uploaded_image_path)
    img = img.resize((input_shape[0], input_shape[1]))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    embedding = siamese_model.predict([img_array, img_array])

    for product_id, product_image in product_mapping.items():
        # Preprocess product image
        product_img = Image.open(product_image)
        product_img = product_img.resize((input_shape[0], input_shape[1]))
        product_img_array = np.array(product_img) / 255.0
        product_img_array = np.expand_dims(product_img_array, axis=0)
        
        product_embedding = siamese_model.predict([product_img_array, product_img_array])
        
        similarity = cosine_similarity(embedding, product_embedding)
        if similarity > 0.8:  # Adjust similarity threshold as needed
            similar_product_info.append({"product_image": product_image, "product_id": product_id})

    return similar_product_info

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the index.html template."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/compare_images')
async def compare_images(image: UploadFile = File(...)):
    """Compare the uploaded image with product images."""
    image_id = str(uuid.uuid4())
    with open(f"uploads/{image_id}.jpg", "wb") as buffer:
        buffer.write(await image.read())

    print("Uploaded image saved successfully:", f"uploads/{image_id}.jpg")

    similar_product_info = process_uploaded_image(image_id, f"uploads/{image_id}.jpg")

    return similar_product_info
