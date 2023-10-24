import tensorflow as tf
import numpy as np
import cv2
import io
from rembg import remove
from PIL import Image

# Replace 'model.tflite' with the path to your TFLite model
main_path = 'converted_tflite_cloth_classification'
model_tflite = f'{main_path}/model_unquant.tflite'

#load labels
with open(f"{main_path}/labels.txt", "r") as f:
    labels = f.read().splitlines()

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=model_tflite)
interpreter.allocate_tensors()

# new
# Abre la imagen y elimina el fondo
image_path = "polo_azul.jpg"
input_image = Image.open(image_path)
output_image = remove(input_image)

# Convierte la imagen a un formato adecuado para OpenCV sin guardarla en disco
output_buffer = io.BytesIO()
output_image.save(output_buffer, format='PNG')
output_buffer.seek(0)

input_image = cv2.imdecode(np.frombuffer(output_buffer.read(), np.uint8), -1)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
input_image = cv2.resize(input_image, (224, 224))
input_image = input_image.astype(np.float32) / 255.0

# Load and preprocess your input image using OpenCV
# image_path = "polo_azul.jpg"
# input_image = cv2.imread(image_path)
# input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Ensure RGB color format
# input_image = cv2.resize(input_image, (224, 224))  # Resize to the expected input size
# input_image = input_image.astype(np.float32)  # Convert to FLOAT32
# input_image /= 255.0  # Normalize to [0, 1] range

# Add batch dimension
input_image = np.expand_dims(input_image, axis=0)

# Get input and output tensors of the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set the input data to the model tensor
interpreter.set_tensor(input_details[0]['index'], input_image)

# Perform inference
interpreter.invoke()

# Get the inference results
results = interpreter.get_tensor(output_details[0]['index'])

# Print the highest probability class
print("Result: " + labels[np.argmax(results[0])])
print("Probability: " + str(np.max(results[0] * 100)))