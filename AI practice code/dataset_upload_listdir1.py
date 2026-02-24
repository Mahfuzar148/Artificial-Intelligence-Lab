import os

import pandas as pd
from sklearn.model_selection import train_test_split

image_paths = []
labels = []

base_path = "dataset"

for img in os.listdir(base_path):

    if img.lower().endswith(('.png','.jpg','.jpeg')):

        image_paths.append(os.path.join(base_path, img))

        # label extract from filename
        label = img.split('_')[0]   # cat_001.jpg → cat
        labels.append(label)

print("Total Images:", len(image_paths))


# Split
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths,
    labels,
    test_size=0.3,
    stratify=labels,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))