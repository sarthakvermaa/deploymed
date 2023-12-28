import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="X-Ray Check!",
    page_icon="👨‍⚕️" 
)

custom_css = """
    body {
        background-color: #222;
        color: #fff;
    }
    .st-b8 {
        background-color: #333; /* Change sidebar color */
    }
    /* Add more styles as needed */
"""

st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)


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

st.markdown(""" 
    <style>
    # #MainMenu {visibility: hidden;}
    # footer {visibility: hidden;} 

    .title {
        text-align: center;
        padding: 5px ;
        font-size: 36px;
        font-weight: bold; /* Adjust font weight */
        color: #2214c7; /* Change text color */
        font-family: 'Calibri', sans-serif; /* Change font family */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .description {
        text-align: center;
        padding: 5px ;
        font-size: 18px;
        margin-bottom: 10px;
    }
    .result-text {
        font-size: 28px;
        margin-bottom: 10px;
        color: #ffffff;
        text-align: center;
    }
    .confidence-text {
        font-size: 18px;
        color: #ffffff;
        text-align: center;
    }
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 10vh; /* Adjust the height as needed */
    }    
    </style>
    """,
    unsafe_allow_html=True
)
tabs = st.sidebar.radio("Navigation", ("Prediction", "BackEnd Used Model Graphs", "About"))

if tabs == "Prediction":
    st.markdown('<h1 class="title">Medical Image Classification WebApp</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">Upload an XRay and let the model help you :)</p>', unsafe_allow_html=True)
    
    st.markdown('<hr>', unsafe_allow_html=True)  # Horizontal line
    
    image_static = Image.open("portrait-doctor.jpg")
    if image_static:
        col1, col2 = st.columns(2)  # Split the screen into two columns
        with col1:
            st.image(image_static, width=300)  # Adjust width as needed
        with col2:
            uploaded_file = st.file_uploader("Upload an X-Ray (JPG)", type="jpg", accept_multiple_files=False)

    if uploaded_file is not None:
        st.markdown('<hr>', unsafe_allow_html=True)  # Horizontal line

        image_uploaded = Image.open(uploaded_file)
        col1, col2 = st.columns(2)  # Split the screen into two columns
        with col1:
            st.image(image_uploaded, width=300, caption="Uploaded Image", use_column_width=True)
        with col2:
            st.markdown('<div style="display: flex; flex-direction: column; height: 100%; justify-content: center; align-items: center;">', unsafe_allow_html=True)
            st.markdown('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
            if st.button('Predict', key='predict_btn'):
                with st.spinner('Predicting...'):
                    prediction = predict(uploaded_file)

                    confidence = abs(prediction - 0.5) * 200
                    confidence_float = float(confidence)  # Convert to float

                    result_text = ""
                    confidence_text = f"My model is {confidence_float:.2f}% confident"

                    if prediction < 0.5:
                        result_text = "NORMAL"
                        box_color = "green"
                    else:
                        result_text = "INFECTIOUS"
                        box_color = "red"

                    # Display result in a styled box
                    st.markdown(
                        f"""
                        <div style="padding: 10px; border-radius: 5px; background-color: {box_color};">
                            <p class="result-text">{result_text}</p>
                            <p class="confidence-text">{confidence_text}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)

elif tabs == "BackEnd Used Model Graphs":
    
    st.markdown('<h1 class="title">Model Stats</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">My Model Graphs</p>', unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)  # Horizontal line
    
    image_1 = Image.open("accuracy_plot (1).jpg")
    image_2 = Image.open("loss_plot (1).jpg")
    image_3 = Image.open("roc_curve.jpg")

    if image_1:
        col1, col2, col3 = st.columns(3)  # Split the screen into two columns
        with col1:
            st.image(image_1, caption="My Model Graph 1", use_column_width=True, width=300)
        with col2:
            st.image(image_2, caption="My Model Graph 2", use_column_width=True, width=300)
        with col3:
            st.image(image_3, caption="My Model Graph 3", use_column_width=True, width=300)
    
    st.markdown('<hr>', unsafe_allow_html=True)

elif tabs == "About":
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<p class="description">This model has been developed as part of the tasks outlined in the OpenCode-23 Medical-Image-Classification <a href="https://github.com/opencodeiiita/Medical-Image-Classification">repository</a>. It aims to create an effective and efficient CNN model capable of categorizing X-ray images into Normal and Infected categories. </p>', unsafe_allow_html=True)
    st.markdown('<p class="description"><br>Made with ❤ by <a href="https://github.com/sarthakvermaa">Sarthak Verma</a>. </p>', unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)