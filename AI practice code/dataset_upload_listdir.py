import os

import pandas as pd
from sklearn.model_selection import train_test_split

# 🔹 Step 1: Collect Image Paths & Labels
image_paths = []
labels = []

base_path = "dataset"

for class_name in os.listdir(base_path):

    class_path = os.path.join(base_path, class_name)

    if not os.path.isdir(class_path):
        continue

    for img in os.listdir(class_path):

        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(class_path, img))
            labels.append(class_name)

print("Total Images:", len(image_paths))


# 🔹 Step 2: First Split (70% Train, 30% Temp)
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths,
    labels,
    test_size=0.3,
    stratify=labels,
    random_state=42
)

# 🔹 Step 3: Second Split (15% Val, 15% Test)
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


# 🔹 Step 4: Convert to DataFrame (for generator use)
train_df = pd.DataFrame({"filename": X_train, "class": y_train})
val_df = pd.DataFrame({"filename": X_val, "class": y_val})
test_df = pd.DataFrame({"filename": X_test, "class": y_test})

print(train_df.head())