import numpy as np
import tensorflow as tf
from PIL import Image
import io

MODEL_PATH = "models/model.tflite"
IMAGE_SIZE = (224, 224)

# Define your classes here in the same order as training
accessory_classes = [
    'Charger',
    'Game-Controller',
    'Headphone',
    'Keyboard',
    'Laptop',
    'Monitor',
    'Mouse',
    'Smartphone',
    'Smartwatch',
    'Speaker'
]

def preprocess_image(image):
    """Preprocess the uploaded image (PIL Image)"""
    img = image.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predict_accessory(uploaded_file, debug=False):
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Read and preprocess image
    if isinstance(uploaded_file, bytes):
        image = Image.open(io.BytesIO(uploaded_file)).convert('RGB')
    else:
        image = Image.open(uploaded_file).convert('RGB')

    input_data = preprocess_image(image)

    # Ensure correct dtype
    if input_details[0]['dtype'] != input_data.dtype:
        input_data = input_data.astype(input_details[0]['dtype'])

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    probabilities = tf.nn.softmax(output).numpy()

    predicted_index = np.argmax(probabilities)
    return accessory_classes[predicted_index], float(probabilities[predicted_index])
