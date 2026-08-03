import os
import random
import shutil

# Original PlantVillage dataset
SOURCE = r"C:\Users\OM SAI\Downloads\archive\PlantVillage"

# Your project dataset folder
DEST = r"C:\Users\OM SAI\Documents\CNN_Plant_Disease_Detection\dataset"

classes = {
    "Potato___Early_blight": "Early_Blight",
    "Potato___Late_blight": "Late_Blight",
    "Potato___healthy": "Healthy"
}

random.seed(42)

for src_folder, dest_folder in classes.items():

    source_path = os.path.join(SOURCE, src_folder)

    images = [
        f for f in os.listdir(source_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    split = int(len(images) * 0.8)

    train_images = images[:split]
    test_images = images[split:]

    train_path = os.path.join(DEST, "train", dest_folder)
    test_path = os.path.join(DEST, "test", dest_folder)

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for img in train_images:
        shutil.copy(
            os.path.join(source_path, img),
            os.path.join(train_path, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(source_path, img),
            os.path.join(test_path, img)
        )

print("✅ Dataset split completed successfully!")