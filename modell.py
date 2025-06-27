import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- Config ---
DATA_DIR = "data"
IMAGE_SIZE = 64

# --- Step 1: Load Data ---
X, y = [], []
class_names = sorted(os.listdir(DATA_DIR))
label_map = {name: idx for idx, name in enumerate(class_names)}

print("Loading images...")
for label in class_names:
    path = os.path.join(DATA_DIR, label)
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            X.append(img)
            y.append(label_map[label])

X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1).astype('float32') / 255.0
y = to_categorical(np.array(y))

# --- Step 2: Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 3: Build Model ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# --- Step 4: Train ---
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# --- Step 5: Save ---
model.save("gesture_model.h5")
print("Model trained and saved as gesture_model.h5 ✅")

