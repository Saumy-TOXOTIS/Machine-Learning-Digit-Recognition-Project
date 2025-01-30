import tensorflow as tf
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("digit_recognition_model.h5")

def preprocess_image(image_path):
    """Preprocess any image to match the model input format."""
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to 28x28 while maintaining aspect ratio
    img = resize_with_padding(img, (28, 28))

    # Normalize pixel values (0-1)
    img = img.astype("float32") / 255.0

    # Reshape for model input (batch_size, height, width, channels)
    img = img.reshape(1, 28, 28, 1)

    return img

def resize_with_padding(image, target_size):
    """Resize the image while maintaining aspect ratio and padding if needed."""
    old_h, old_w = image.shape[:2]
    target_h, target_w = target_size

    # Compute the scaling factor
    scale = min(target_w / old_w, target_h / old_h)
    new_w = int(old_w * scale)
    new_h = int(old_h * scale)

    # Resize the image with the computed scaling factor
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a blank canvas (black background)
    padded_img = np.zeros((target_h, target_w), dtype=np.uint8)

    # Compute padding
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    # Place the resized image onto the blank canvas
    padded_img[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_img

    return padded_img

def predict_digit(image_path):
    """Predict the digit from an image file."""
    # Preprocess the image
    img = preprocess_image(image_path)

    # Predict using the model
    prediction = model.predict(img)
    digit = np.argmax(prediction)

    return digit

# Example usage
image_path = "test_digit.png"  # Replace with your image
predicted_digit = predict_digit(image_path)
print(f"Predicted Digit: {predicted_digit}")