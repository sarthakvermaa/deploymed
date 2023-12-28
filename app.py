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

# Applying CSS styles
st.markdown(
    """
    <style>
    .title {
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
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .result-text {
        font-size: 28px;
        margin-bottom: 10px;
        color: #333333;
        text-align: center;
    }
    .confidence-text {
        font-size: 18px;
        color: #555555;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

tabs = st.sidebar.radio("Navigation", ("Prediction", "BackEnd Used Model Graphs"))

if tabs == "Prediction":
    st.markdown('<hr>', unsafe_allow_html=True)  # Horizontal line
    
    st.markdown('<h1 class="title">Medical Image Classification WebApp</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">Upload an XRay and let the model help you :) </p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type="jpg", accept_multiple_files=False)

    if uploaded_file is not None:
        st.markdown('<hr>', unsafe_allow_html=True)  # Horizontal line

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button('Predict', key='predict_btn'):
            with st.spinner('Predicting...'):
                prediction = predict(uploaded_file)
                st.markdown('<div class="upload-container">', unsafe_allow_html=True)
                
                confidence = abs(prediction - 0.5) * 200
                confidence_float = float(confidence)  # Convert to float
                
                result_text = ""
                confidence_text = f"My model is {confidence_float:.2f}% confident"
                
                if prediction < 0.5:
                    result_text = "NORMAL"
                else:
                    result_text = "INFECTIOUS"
                
                # Display result with styling
                st.markdown(f'<p class="result-text">{result_text}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="confidence-text">{confidence_text}</p>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

elif tabs == "BackEnd Used Model Graphs":
    st.markdown('<hr>', unsafe_allow_html=True)  # Horizontal line
    
    st.markdown('<h1 class="title">Model Stats</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">My Model Graphs</p>', unsafe_allow_html=True)

    image_1 = Image.open("accuracy_plot (1).jpg")
    image_2 = Image.open("loss_plot (1).jpg")
    image_3 = Image.open("roc_curve.jpg")

    st.image(image_1, caption="My Model Graph 1", use_column_width=True)
    st.image(image_2, caption="My Model Graph 2", use_column_width=True)
    st.image(image_3, caption="My Model Graph 3", use_column_width=True)
