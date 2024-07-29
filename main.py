import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from util import classify, set_background

set_background('./bgs/22.jpg')

# Custom CSS for styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

        body {
            font-family: 'Roboto', sans-serif;
            background-color: #282c34;
        }

        h1 {
            color: white;
            font-size: 3em;
            text-align: center;
            margin-top: 0.5em;
        }

        h2 {
            color: white;
            font-size: 1.5em;
            text-align: center;
            margin-bottom: 1em;
        }

        .stFileUploader {
            display: flex;
            justify-content: center;
            margin-bottom: 2em;
        }

        .stImage {
            display: flex;
            justify-content: center;
            margin-bottom: 2em;
        }

        h3 {
            color: white;
            font-size: 1.75em;
            font-weight: bold;
            text-align: center;
        }

        h4 {
            color: white;
            font-size: 1.25em;
            text-align: center;
        }
        
        .uploaded-file-label {
            color: white;
            font-size: 1.25em;
            text-align: center;
            margin-top: 1em;
        }
    </style>
""", unsafe_allow_html=True)

# Set title with white text color
st.markdown("<h1>Pneumonia Classification</h1>", unsafe_allow_html=True)

# Set header with white text color
st.markdown("<h2>Please upload a chest X-ray image</h2>", unsafe_allow_html=True)

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png', 'gif'])

# Load classifier
model = load_model('./model/pneumonia_classifier.h5')

# Load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# Display image and label
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)
    st.markdown(f"<div class='uploaded-file-label'>Uploaded Image</div>", unsafe_allow_html=True)

    # Classify image
    class_name, conf_score = classify(image, model, class_names)

    # Write classification with white text color and bold formatting
    st.markdown(f"<h3>Prediction: {class_name}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h4>Score: {int(conf_score * 1000) / 10}%</h4>", unsafe_allow_html=True)
