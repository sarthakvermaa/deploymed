import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model = tf.keras.models.load_model('resnet_model.h5')

# Function to make predictions
def predict(image):
    img = Image.open(image).convert('RGB')  
    img = img.resize((224, 224))  
    img_array = np.array(img)  
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    predictions = model.predict(img_array)

    return predictions
with st.sidebar:
    tabs = st.radio("Navigation", ("Prediction", "BackEnd Used Model Graphs"))
# New layout and style
st.markdown(
    """
    <style>
    body {
        background-color: #f9f9f9;
        font-family: Arial, sans-serif;
    }
    .title {
        color: #333333;
        text-align: center;
        padding: 20px 0;
        font-size: 36px;
    }
    .description {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .upload-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        margin: 0 auto;
        max-width: 600px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .upload-text {
        font-size: 24px;
        margin-bottom: 15px;
    }
    .predict-button {
        padding: 10px 20px;
        font-size: 18px;
        border-radius: 5px;
        background-color: #4CAF50;
        color: white;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .predict-button:hover {
        background-color: #45a049;
    }
    .prediction-container {
        text-align: center;
        margin-top: 30px;
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        max-width: 600px;
        margin: 0 auto;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .prediction-header {
        font-size: 28px;
        margin-bottom: 20px;
        color: #333333;
    }
    .prediction-results {
        font-size: 18px;
        color: #555555;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if tabs == "Prediction":
    st.markdown('<h1 class="title">Medical Image Classification WebApp</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">Upload an XRay and let the model help you :) </p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type="jpg", accept_multiple_files=False)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button('Predict', key='predict_btn'):
            with st.spinner('Predicting...'):
                prediction = predict(uploaded_file)
                st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
                st.markdown('<p class="prediction-header">Prediction Results:</p>', unsafe_allow_html=True)
                
                confidence = abs(prediction - 0.5) * 200
                confidence_float = float(confidence)  # Convert to float
                
                if prediction < 0.5:
                    st.write("NORMAL")
                    st.write(f"My model is {confidence_float:.2f}% confident")
                else:
                    st.write("INFECTIOUS")
                    st.write(f"My model is {confidence_float:.2f}% confident")
                
                st.markdown('</div>', unsafe_allow_html=True)

elif tabs == "BackEnd Used Model Graphs":
    st.markdown('<h1 class="title">Model Stats</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">My Model Graphs</p>', unsafe_allow_html=True)

    image_1 = Image.open("accuracy_plot (1).jpg")
    image_2 = Image.open("loss_plot (1).jpg")
    image_3 = Image.open("roc_curve.jpg")

    st.image(image_1, caption="My Model Graph 1", use_column_width=True)
    st.image(image_2, caption="My Model Graph 2", use_column_width=True)
    st.image(image_3, caption="My Model Graph 3", use_column_width=True)
