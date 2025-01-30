import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Added for augmentation
import struct
import os
import cv2

# Function to load correction images from the correction dataset folder
def load_correction_images(correction_folder, target_size=(28, 28)):
    """Loads manually corrected images and labels"""
    images = []
    labels = []

    for digit_label in range(10):  # Loop over possible digits (0-9)
        digit_folder = os.path.join(correction_folder, str(digit_label))
        if not os.path.exists(digit_folder):
            continue  # Skip if folder doesn't exist

        for filename in os.listdir(digit_folder):
            img_path = os.path.join(digit_folder, filename)

            # Read image and convert to grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Resize to 28x28 while keeping aspect ratio
            img = resize_with_padding(img, target_size)

            # Normalize pixel values (0-1)
            img = img.astype("float32") / 255.0

            # Append to dataset
            images.append(img)
            labels.append(digit_label)

    print(f"Loaded {len(images)} correction images from {correction_folder}")
    return np.array(images), np.array(labels)

def resize_with_padding(image, target_size):
    """Resizes an image while maintaining aspect ratio and adding padding"""
    old_h, old_w = image.shape[:2]
    target_h, target_w = target_size

    scale = min(target_w / old_w, target_h / old_h)
    new_w = int(old_w * scale)
    new_h = int(old_h * scale)

    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded_img = np.zeros((target_h, target_w), dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    padded_img[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_img

    return padded_img

# Function to load IDX files
def load_idx_images(filename):
    """ Load images from IDX3-UBYTE file """
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28, 28)
    return images

def load_idx_labels(filename):
    """ Load labels from IDX1-UBYTE file """
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Load MNIST dataset from IDX files
train_images = load_idx_images("train-images.idx3-ubyte")
train_labels = load_idx_labels("train-labels.idx1-ubyte")
test_images = load_idx_images("t10k-images.idx3-ubyte")
test_labels = load_idx_labels("t10k-labels.idx1-ubyte")

# Normalize images to range [0,1]
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Reshape images to match CNN input
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Convert labels to one-hot encoding
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

# Load correction dataset (if available)
correction_images, correction_labels = load_correction_images("correction_dataset")

if len(correction_images) > 0:
    # Reshape correction images
    correction_images = correction_images.reshape(-1, 28, 28, 1)

    # Convert correction labels to categorical (one-hot encoding)
    correction_labels = keras.utils.to_categorical(correction_labels, 10)

    # Merge MNIST dataset with correction dataset
    train_images = np.concatenate((train_images, correction_images), axis=0)
    train_labels = np.concatenate((train_labels, correction_labels), axis=0)

    print(f"Added {len(correction_images)} new images to training dataset!")
    print(f"🔹 New Training Data Size: {train_images.shape[0]} images")

# Data augmentation (use ImageDataGenerator)
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(train_images)

# Define the CNN model
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Use learning rate scheduler to adjust the learning rate during training
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-3 * 0.9 ** epoch, verbose=1
)

# Train the model with data augmentation
model.fit(datagen.flow(train_images, train_labels, batch_size=128), 
          epochs=20, 
          validation_data=(test_images, test_labels),
          callbacks=[lr_schedule])

# Save the trained model
model.save("digit_recognition_model.h5")

print("Model training complete. Saved as 'digit_recognition_model.h5'")