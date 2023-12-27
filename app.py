import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('resnet_model.h5')

def predict(image):
    img = Image.open(image).convert('RGB')  
    img = img.resize((224, 224))  
    img_array = np.array(img)  
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    predictions = model.predict(img_array)

    return predictions

st.title("Deep Learning Model Deployment")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    
    result = predict(uploaded_file)

    st.write("Prediction Results:")
    st.write(result)
