import os
import tensorflow as tf
import streamlit as st
import numpy as np

st.header('Expression Classification CNN Model')

expression_names = ['marah', 'takut', 'senang', 'netral', 'sedih']

# Load the model with the new .keras format
model = tf.keras.models.load_model('best_model2.keras')

def preprocess_image(image_path):
    # Load image
    img = tf.keras.utils.load_img(image_path, target_size=(48, 48), color_mode='grayscale')
    
    # Convert to array
    img_array = tf.keras.utils.img_to_array(img)
    
    # Preprocessing
    img_array = img_array / 255.0  # Normalize
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# def classify_images(image_path):
#     try:
#         # Preprocess image
#         processed_image = preprocess_image(image_path)
        
#         # Make prediction with confidence threshold
#         predictions = model.predict(processed_image)
#         predicted_class = np.argmax(predictions[0])
#         confidence = predictions[0][predicted_class] * 100
        
#         # Only show prediction if confidence is above threshold
#         if confidence < 50:  # Adjust threshold as needed
#             return "Tidak dapat menentukan ekspresi dengan keyakinan yang cukup"
            
#         # Langsung kembalikan ekspresi dan persen keyakinan
#         return f"{expression_names[predicted_class]} ({confidence:.2f}%)"
#     except Exception as e:
#         return f'Error dalam klasifikasi gambar: {e}'
def classify_images(image_path):
    try:
        # Preprocess image
        processed_image = preprocess_image(image_path)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100
        
        # Langsung kembalikan ekspresi dan persen keyakinan
        return f"{expression_names[predicted_class]} ({confidence:.2f}%)"
    except Exception as e:
        return f'Error dalam klasifikasi gambar: {e}'


uploaded_file = st.file_uploader('Unggah Gambar', type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    upload_folder = 'temp'  # Pastikan folder 'temp' ada atau dibuat
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    file_path = os.path.join(upload_folder, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, width=200)
    st.markdown(classify_images(file_path))

    # Hapus file temporary setelah digunakan
    os.remove(file_path)