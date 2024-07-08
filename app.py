import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image

# Load your trained model
loaded_model = load_model('model.h5')

# Define class names
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

def get_prediction(image):
    # Convert PIL image to OpenCV format
    image = np.array(image)

    # Resize the image to the target size
    test_image = cv2.resize(image, (130, 130))

    # Convert the image from RGB to BGR format (if needed)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

    # Normalize the image data if your model was trained with normalized data
    test_image = test_image / 255.0

    # Add an extra dimension to the image to represent the batch size
    test_image = np.expand_dims(test_image, axis=0)

    # Predict using the model
    prediction = loaded_model.predict(test_image)

    # Get the predicted class
    predicted_class = np.argmax(prediction, axis=1)

    return predicted_class[0]

# Streamlit app
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")
st.title("🧠 Brain Tumor Detection")

# Display banner image below the title
st.image("banner.png", use_column_width=True)

st.write("Upload an MRI image to predict the type of brain tumor.")

# Display logo in the sidebar
st.sidebar.image("MICROAI1.JPG", use_column_width=True)

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    # st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button("Predict"):
        st.markdown("### Prediction")
        with st.spinner('Analyzing the image...'):
            # Make prediction
            predicted_class = get_prediction(image)

            # Display the image on the left and prediction on the right
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption='Uploaded Image.', use_column_width=True)
            with col2:
                st.success(f"**Predicted Class Name:** {class_names[predicted_class]}")
else:
    st.info("Please upload a JPG, JPEG or PNG image to get a prediction.")
