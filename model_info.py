from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Load the model
model = load_model("digit_recognition_model.h5")

# Print model summary
model.summary()

# Get all layer weights
for layer in model.layers:
    print(layer.name)
    print(layer.get_weights())

# Visualize the model's architecture
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)