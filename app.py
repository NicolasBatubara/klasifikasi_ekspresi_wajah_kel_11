import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np

st.header('Expression Classification CNN Model')

expression_names = ['marah', 'sedih', 'senang']

model = load_model('Face_Recog_Model.h5')

def classify_images(image_path):
    try:
        # Load and preprocess image
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = np.expand_dims(input_image_array, axis=0)

        # Predict and apply softmax
        predictions = model.predict(input_image_exp_dim)
        probabilities = tf.nn.softmax(predictions[0])  # Apply softmax to convert logits to probabilities

        # Get class with highest probability
        predicted_class = np.argmax(probabilities)
        predicted_confidence = probabilities[predicted_class] * 100  # Convert to percentage

        # Format output with two decimal places
        outcome = f'Gambar ini termasuk dalam kelas "{expression_names[predicted_class]}" dengan skor {predicted_confidence:.2f}%'
        return outcome
    except Exception as e:
        return f'Error dalam klasifikasi gambar: {e}'

uploaded_file = st.file_uploader('Unggah Gambar', type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    upload_folder = 'upload'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    file_path = os.path.join(upload_folder, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, width=200)
    st.markdown(classify_images(file_path))
