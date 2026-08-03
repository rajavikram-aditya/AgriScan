import os
import shutil
import random
import kagglehub

def process_dataset_dir(source_dir, dest_train, dest_val, crop_prefix=None):
    """
    Recursively find directories containing images, and copy/split them.
    Limits each class to a maximum of 1000 images to balance the dataset.
    """
    for root, dirs, files in os.walk(source_dir):
        images = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        if len(images) > 0:
            folder_name = os.path.basename(root)
            
            # Normalize folder name
            dest_folder = folder_name.replace(" ", "_")
            if crop_prefix:
                # If it doesn't already have the prefix, add it
                if not dest_folder.lower().startswith(crop_prefix.lower()):
                    dest_folder = f"{crop_prefix}___{dest_folder}"
                else:
                    # e.g., if it's "Tomato___Bacterial_spot" it's fine.
                    pass
            
            # Filter logic based on the required crops
            valid_prefixes = ["Tomato___", "Corn_", "Rice___", "Wheat___", "Potato___", "Sugarcane___", "Cotton___"]
            if not any(dest_folder.startswith(p) for p in valid_prefixes):
                continue
                
            random.shuffle(images)
            
            # Cap the number of images per class to 1000
            if len(images) > 1000:
                images = images[:1000]
                
            split = int(len(images) * 0.8)
            train_images = images[:split]
            val_images = images[split:]
            
            train_path = os.path.join(dest_train, dest_folder)
            val_path = os.path.join(dest_val, dest_folder)
            
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(val_path, exist_ok=True)
            
            print(f"Processing {dest_folder}: {len(train_images)} train, {len(val_images)} val")
            
            for img in train_images:
                shutil.copy(os.path.join(root, img), os.path.join(train_path, img))
            
            for img in val_images:
                shutil.copy(os.path.join(root, img), os.path.join(val_path, img))


def main():
    random.seed(42)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dest_train = os.path.join(base_dir, 'dataset', 'train')
    dest_val = os.path.join(base_dir, 'dataset', 'val')
    
    # 1. Handle existing Potato Dataset from Folder 1
    # We will look in the original directory if it exists relative to AgriScan
    original_potato_train = os.path.join(base_dir, '..', 'CNN-Plant-Disease-Detection-main', 'dataset', 'train')
    original_potato_test = os.path.join(base_dir, '..', 'CNN-Plant-Disease-Detection-main', 'dataset', 'test')
    
    potato_mapping = {
        "Early_Blight": "Potato___Early_blight",
        "Late_Blight": "Potato___Late_blight",
        "Healthy": "Potato___healthy"
    }
    
    if os.path.exists(original_potato_train):
        print("Migrating original Potato dataset...")
        for old_name, new_name in potato_mapping.items():
            for src_base, dest_base in [(original_potato_train, dest_train), (original_potato_test, dest_val)]:
                src_path = os.path.join(src_base, old_name)
                dest_path = os.path.join(dest_base, new_name)
                if os.path.exists(src_path):
                    os.makedirs(dest_path, exist_ok=True)
                    images = [f for f in os.listdir(src_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                    if len(images) > 1000 and dest_base == dest_train:
                        images = images[:800] # roughly 80% of 1000
                    for img in images:
                        shutil.copy(os.path.join(src_path, img), os.path.join(dest_path, img))
                    print(f"Migrated {len(images)} images for {new_name} to {dest_base}")
    
    # 2. Download and process Kaggle Datasets
    print("Downloading emmarex/plantdisease (Tomato + Corn)...")
    path1 = kagglehub.dataset_download("emmarex/plantdisease")
    process_dataset_dir(path1, dest_train, dest_val)
    
    print("Downloading anshulm257/rice-disease-dataset (Rice)...")
    path2 = kagglehub.dataset_download("anshulm257/rice-disease-dataset")
    process_dataset_dir(path2, dest_train, dest_val, crop_prefix="Rice")
    
    print("Downloading khanaamer/wheat-leaf-disease-dataset (Wheat)...")
    path3 = kagglehub.dataset_download("khanaamer/wheat-leaf-disease-dataset")
    process_dataset_dir(path3, dest_train, dest_val, crop_prefix="Wheat")

    print("Downloading nirmalsankalana/sugarcane-leaf-disease-dataset (Sugarcane)...")
    path4 = kagglehub.dataset_download("nirmalsankalana/sugarcane-leaf-disease-dataset")
    process_dataset_dir(path4, dest_train, dest_val, crop_prefix="Sugarcane")

    print("Downloading seroshkarim/cotton-leaf-disease-dataset (Cotton)...")
    path5 = kagglehub.dataset_download("seroshkarim/cotton-leaf-disease-dataset")
    process_dataset_dir(path5, dest_train, dest_val, crop_prefix="Cotton")

    print("\nDataset splitting and merging completed successfully!")
    
    # 3. Print final class list and counts
    print("\n--- Final Dataset Summary ---")
    total_train = 0
    total_val = 0
    classes = sorted(os.listdir(dest_train))
    for cls in classes:
        train_count = len(os.listdir(os.path.join(dest_train, cls)))
        val_count = len(os.listdir(os.path.join(dest_val, cls))) if os.path.exists(os.path.join(dest_val, cls)) else 0
        total_train += train_count
        total_val += val_count
        print(f"{cls}: Train={train_count}, Val={val_count}, Total={train_count + val_count}")
        
    print(f"\nTotal Classes: {len(classes)}")
    print(f"Total Train Images: {total_train}")
    print(f"Total Val Images: {total_val}")

if __name__ == "__main__":
    main()
