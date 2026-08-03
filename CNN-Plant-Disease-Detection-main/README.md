# CNN Plant Disease Detection

## Project Overview

CNN Plant Disease Detection is a Deep Learning based project that detects plant diseases from leaf images.

The model uses a Convolutional Neural Network (CNN) to classify leaf images into three categories:

- Healthy
- Early Blight
- Late Blight

The main objective of this project is to identify plant diseases automatically using Artificial Intelligence and Computer Vision.

## Technologies Used

- Python
- TensorFlow
- Keras
- CNN (Convolutional Neural Network)
- Streamlit
- NumPy
- OpenCV

## Dataset

The dataset contains plant leaf images divided into:

- Training Dataset
- Testing Dataset

Classes:

1. Healthy
2. Early_Blight
3. Late_Blight

## Model Description

The CNN model learns important features from plant leaf images and classifies them into different disease categories.

Process:

- Image resizing to 224 × 224 pixels
- Feature extraction using CNN layers
- Disease classification
- Confidence percentage prediction

## Project Structure

```
CNN_Plant_Disease_Detection

├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── plant_disease_model.keras
├── README.md

└── dataset
    ├── train
    └── test
```
## How to Run Project
### Step 1: Install Required Libraries

Open terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

### Step 2: Run Streamlit Application

Run the following command:

```bash
streamlit run app.py
```

### Step 3: Use the Application

- Browser will open automatically.
- Upload a plant leaf image.
- The CNN model will analyze the image.
- The application will display:
  - Predicted disease class
  - Confidence percentage

## Features

- Upload plant leaf images
- Automatic disease detection
- CNN-based classification
- Confidence score display
- User-friendly Streamlit interface

## Model Performance

Testing Results:

| Disease Class | Confidence |
|---|---|
| Healthy | 94.28% |
| Late Blight | 99.86% |
| Early Blight | 100% |

## Future Improvements

- Add more plant species
- Add more disease categories
- Increase dataset size
- Improve model accuracy
- Deploy application online
- Add real-time camera detection

## Conclusion

This project demonstrates the use of Deep Learning and Computer Vision for automatic plant disease detection.

The system can help in early identification of plant diseases and support better agricultural practices.