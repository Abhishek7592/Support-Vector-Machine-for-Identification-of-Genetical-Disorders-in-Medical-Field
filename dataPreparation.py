import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from config import IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, DISORDER_MAP, CLASSES

def preprocess_image(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT), grayscale=(IMG_CHANNELS == 1)):
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)
        if grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        img_array = np.array(img)
        img_array = img_array / 255.0
        flattened_array = img_array.flatten()
        return flattened_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def prepare_image_data(base_dir="image_data_synthetic", num_samples_per_class=None, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    if num_samples_per_class is None:
        num_samples_per_class = {0: 100, 1: 20, 2: 15, 3: 25}

    all_images_flattened = []
    all_labels = []

    if not os.path.exists(base_dir) or len(glob.glob(os.path.join(base_dir, "*", "*.png"))) == 0:
        print(f"Synthetic data directory '{base_dir}' not found or empty. Creating placeholders...")
        if not os.path.exists(base_dir): os.makedirs(base_dir)
        for class_idx, class_name_key in enumerate(CLASSES):
             safe_folder_name = f"class_{class_idx}_{DISORDER_MAP[class_name_key].split(' ')[0].lower()}"
             class_dir = os.path.join(base_dir, safe_folder_name)
             if not os.path.exists(class_dir): os.makedirs(class_dir)

             num_images = num_samples_per_class.get(class_idx, 10)
             print(f"  Creating {num_images} placeholder images for class {class_idx} in '{class_dir}'")
             for i in range(num_images):
                 img_array = np.ones((target_size[1], target_size[0]), dtype=np.uint8) * (50 + class_idx * 50)
                 img = Image.fromarray(img_array)
                 img_path = os.path.join(class_dir, f"placeholder_{class_idx}_{i}.png")
                 img.save(img_path)
        print("Placeholder image structure created.")

    print(f"Loading image data from '{base_dir}'...")
    found_images = False
    for class_idx, class_name_key in enumerate(CLASSES):
        safe_folder_name = f"class_{class_idx}_{DISORDER_MAP[class_name_key].split(' ')[0].lower()}"
        class_dir = os.path.join(base_dir, safe_folder_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Directory not found for class {class_idx}: {class_dir}")
            continue

        image_files = glob.glob(os.path.join(class_dir, "*.png")) + \
                      glob.glob(os.path.join(class_dir, "*.jpg")) + \
                      glob.glob(os.path.join(class_dir, "*.jpeg"))

        print(f"  Found {len(image_files)} images for class {class_idx} in '{os.path.basename(class_dir)}'")
        for img_path in image_files:
            flattened_img = preprocess_image(img_path, target_size=target_size, grayscale=(IMG_CHANNELS==1))
            if flattened_img is not None:
                all_images_flattened.append(flattened_img)
                all_labels.append(class_idx)
                found_images = True

    if not found_images:
         raise FileNotFoundError(f"No valid images found in the subdirectories of '{base_dir}'. Please check the structure and image files.")

    X = np.array(all_images_flattened)
    y = np.array(all_labels)
    print(f"Loaded data shape: X={X.shape}, y={y.shape}")
    print("Class distribution in loaded data:\n", pd.Series(y).value_counts().sort_index())

    return X, y
