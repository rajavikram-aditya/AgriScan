import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("plant_disease_model.keras")

# Class names
classes = ["Early_Blight", "Healthy", "Late_Blight"]

# Image path
img_path = "test.jpg"   # Prediction ke liye image ka naam

# Load image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)
predicted_class = classes[np.argmax(prediction)]

print("Predicted Disease:", predicted_class)