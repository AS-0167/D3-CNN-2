import numpy as np
from tqdm import tqdm  
import os
import cv2

from configurations import IMG_SIZE, CLASSES, CLASS_TO_IDX

def load_data(data_dir):
    X, y = [], []
    print(f"Loading data from: {data_dir}")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    for label in tqdm(CLASSES, desc="Loading classes"):
        class_dir = os.path.join(data_dir, label)
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Class directory not found: {class_dir}")

        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for fname in tqdm(image_files, desc=f"Loading {label}", leave=False):
            img_path = os.path.join(class_dir, fname)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # First resize to 224x224
                img = cv2.resize(img, (224, 224))
                # Then resize to 32x32
                img = cv2.resize(img, (32, 32))
                
                # Skip if image doesn't have the expected shape
                if img.shape != (32, 32, 3):
                    continue

                X.append(img)
                y.append(CLASS_TO_IDX[label])
            except Exception as e:
                continue

    if len(X) == 0:
        raise ValueError("No images were successfully loaded!")

    # Convert to numpy array and normalize
    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y)

    print(f"Successfully loaded {len(X)} images")
    return X, y