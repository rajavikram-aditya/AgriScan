# AgriScan

AgriScan is a unified plant and crop disease detection project utilizing MobileNetV2 transfer learning.

## Features
- **Crop Types:** Detects diseases in Fruit/Vegetable crops (Potato, Tomato, Cotton, Sugarcane) and Grain Crops (Corn, Rice, Wheat).
- **Deep Learning Model:** Powered by MobileNetV2 for fast and accurate classification across 22 disease classes.
- **Disease Information:** Provides detailed descriptions and treatments for identified diseases.
- **Web App:** Easy-to-use Streamlit web interface with an expandable sidebar showing the full 9-stage image processing pipeline.
- **Shared Preprocessing:** All preprocessing logic lives in `preprocessing.py` -- used by both the app and the pipeline view so they never drift apart.

## Setup Instructions

1. **Install Dependencies:**
   Make sure you have Python 3.8+ installed. Then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download & Process Datasets:**
   Run the dataset setup script to automatically download the Corn, Tomato, Rice, Wheat, Cotton, and Sugarcane datasets from Kaggle, and merge them with the existing Potato dataset:
   ```bash
   python split_dataset.py
   ```

3. **Train the Model:**
   Train the MobileNetV2 model using the combined dataset.
   ```bash
   python train.py
   ```
   This will create `agriscan_model.keras` and `class_indices.json`.

4. **Run the Application:**
   Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
   Toggle the **"Enable Pipeline View"** checkbox in the sidebar to expand all 9 preprocessing stages.

## Folder Structure
- `dataset/` -- Contains `train/` and `val/` subdirectories with `Crop___Disease` class folders.
- `preprocessing.py` -- Shared image preprocessing module.
- `train.py` -- Script to train the CNN using transfer learning.
- `app.py` -- Streamlit application (prediction + expandable pipeline sidebar).
- `image_processing_view.py` -- Standalone pipeline visualization (optional, for dual-screen demos).
- `disease_info.py` -- Database of diseases, descriptions, and treatments.
- `split_dataset.py` -- Script to organize and split the raw image datasets.
