import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Ensure the correct TensorFlow version (2.x required for this script)
print(tf.__version__)

# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    # Load the image file, resizing it to 227x227 pixels (as required by AlexNet)
    img = image.load_img(img_path, target_size=(227, 227))
    # Convert the image to a numpy array and scale the pixel values to [0, 1]
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to predict the dust density in the image
def predict_dust_density(model0, img_array):
    # Use the model to predict the image's class
    prediction = model.predict(img_array)
    # Return the predicted class based on the threshold
    if prediction[0] > 0.5:
        return "Low dust density"
    else:
        return "High dust density"

# Load your pre-trained model
model = load_model('dust_density_model.h5')

# Path to the image you want to classify
test_image_path = 'test_image.jpg'

# Load and preprocess the image
test_image = load_and_preprocess_image(test_image_path)

# Predict the dust density
dust_density_prediction = predict_dust_density(model, test_image)

# Print the prediction
print(f"The image has {dust_density_prediction}")
