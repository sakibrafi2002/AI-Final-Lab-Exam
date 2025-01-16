import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to a scale of 0 to 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the MLP model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into a 1D array
    Dense(128, activation='relu'),  # First hidden layer with 128 neurons
    Dense(64, activation='relu'),   # Second hidden layer with 64 neurons
    Dense(10, activation='softmax') # Output layer with 10 neurons for 10 classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    verbose=2
)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy:.2f}")

# Save the trained model
model.save('digit_recognition_model.h5')

# Function to load and preprocess a custom image
def preprocess_custom_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img)  # Convert to numpy array
    img_array = 255 - img_array  # Invert colors (white digit on black background)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = img_array.reshape(1, 28, 28)  # Reshape for the model
    return img_array

# Load the saved model
loaded_model = load_model('digit_recognition_model.h5')

# Test with a custom image
custom_image_path = 'C:\\aaa\\Varsity\\AI\\Project\\test3.png'  # Replace with your image path
custom_image = preprocess_custom_image(custom_image_path)

# Make a prediction
prediction = loaded_model.predict(custom_image)
predicted_digit = np.argmax(prediction)

print(f"The model predicts this digit is: {predicted_digit}")

# Visualize the custom image and prediction
plt.imshow(Image.open(custom_image_path).convert('L'), cmap='gray')
plt.title(f"Predicted: {predicted_digit}")
plt.axis('off')
plt.show()