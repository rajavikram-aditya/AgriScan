import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_disease_model.keras")
    return model

model = load_model()

# Classes
class_names = [
    "Early_Blight",
    "Healthy",
    "Late_Blight"
]

# Title section
st.title("🌱 Plant Disease Detection")
st.write("Upload a plant leaf image and get disease prediction using CNN.")

st.divider()

# Upload image
uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # Display image
    st.subheader("Uploaded Leaf Image")
    st.image(image, width=350)

    # Prediction button
    if st.button("🔍 Predict Disease"):

        img = image.resize((224, 224))
        img_array = np.array(img)

        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]

        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        disease = class_names[index]

        st.divider()

        st.subheader("Prediction Result")

        if disease == "Healthy":
            st.success(f"🌿 {disease}")
        else:
            st.warning(f"⚠️ {disease}")

        st.info(f"Confidence: {confidence:.2f}%")

st.divider()

st.caption("CNN based Plant Disease Detection System")