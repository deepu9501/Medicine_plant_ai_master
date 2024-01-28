from flask import Flask, request, jsonify
import tensorflow as tf
import os
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import requests
from io import BytesIO

main_data_dir = os.getcwd() + "/Segmented Medicinal Leaf Images"
batch_size = 3
num_classes = len(os.listdir(main_data_dir))
epochs = 10

split_ratio = 0.8
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=1 - split_ratio  # Set validation split
)

train_generator = train_datagen.flow_from_directory(
    main_data_dir,
    target_size=(244, 244),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Specify training subset
)
validation_generator = train_datagen.flow_from_directory(
    main_data_dir,
    target_size=(244, 244),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Specify validation subset

)


def load_and_preprocess_image(image_path):
    # Load the image using cv2
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize to the desired input size (e.g., 224x224)
    image = cv2.resize(image, (224, 224))
    # Normalize the pixel values and preprocess for MobileNetV2
    preprocessed_image = preprocess_input(image)

    return preprocessed_image


# Example usage:
image_path = os.getcwd() + '/leaf2.jpg'  # Replace with the path to your image
preprocessed_image = load_and_preprocess_image(image_path)

# base_model = MobileNetV2(
#     weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(512, activation='relu')(x)
# x = Dropout(0.5)(x)  # Adding dropout for regularization
# predictions = Dense(num_classes, activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=predictions)
# for layer in base_model.layers:
#     layer.trainable = False
# model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# # trainig the model
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples)


# model.save('plant_identification_model2.h5')
base_url = 'http://192.168.56.1:3000/uploads/'
image_path = 'http://192.168.56.1:3000/uploads/asd.jpg'

model = tf.keras.models.load_model('plant_identification_model2.h5')

# Load and preprocess the image


def preprocess_image(image_path):
    # Load the image using cv2
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    image = cv2.resize(image, (224, 224))  # Resize to the desired input size
    image_array = np.expand_dims(image, axis=0)
    preprocessed_image = preprocess_input(image_array)
    return preprocessed_image


def download_image(url, save_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)
            print(f"Image downloaded successfully to {save_path}")
        else:
            print(
                f"Failed to download image. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading image: {str(e)}")

# Perform prediction


def predict_plant(image_path, label_mapping):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)

    # Map model's numeric predictions to labels
    predicted_label_index = np.argmax(predictions)
    predicted_label = label_mapping[predicted_label_index]
    confidence = predictions[0][predicted_label_index]

    return predicted_label, confidence


global label_mapping
label_mapping = {i: label for i, label in enumerate(
    sorted(os.listdir(main_data_dir)))}
# Provide the path to the image you want to classify
# Replace with the path to your image
image_path = './leaf.jpg'
predicted_label, confidence = predict_plant(image_path, label_mapping)

# Print the prediction
print(f"Predicted Label: {predicted_label}")
print(f"Confidence: {confidence:.2f}")

# Check if confidence is 0.15
if confidence == 0.15:
    print("not a plant")

app = Flask(__name__)


@app.route('/api/process_capture', methods=['POST'])
def process_capture():

    data = request.get_json()
    image_url = data.get('imageUrl')
    print(image_url)
    print(data['imgName']['imageName'])
    imgName = data['imgName']['imageName']
    url = base_url+imgName
    response = requests.get(url)
    with open("img_capture.jpeg", "wb") as f:
        f.write(response.content)

    predicted_label, confidence = predict_plant(
        f"./img_capture.jpeg", label_mapping)
    print("N", predicted_label, confidence)
    # Add your image processing code here
    # You can run your functions on the image URL and generate a JSON response
    print(type(confidence))
    print(type(predicted_label))
    sci_name, name = predicted_label.split("(")
    # For demonstration purposes, we'll return a JSON response with status code 200
    response_data = {
        'name': name[:-1],
        'sci_name': sci_name,
        'confidence': str(confidence),
    }

    return jsonify(response_data)


@app.route('/api/process_submit', methods=['POST'])
def process_submit():

    data = request.get_json()
    image_url = data.get('imageUrl')
    print(data['imgUrl']['imageUrl'])
    imgName = data["imgName"]["imageName"]
    imgUrl = data['imgUrl']['imageUrl']
    print(imgName, imgUrl)
    image_url = imgUrl
    save_path = f"./{imgName}"
    download_image(image_url, save_path)

    predicted_label, confidence = predict_plant(f"./{imgName}", label_mapping)
    print("N", predicted_label, confidence)
    # Add your image processing code here
    # You can run your functions on the image URL and generate a JSON response
    print(type(confidence))
    print(type(predicted_label))
    sci_name, name = predicted_label.split("(")
    # For demonstration purposes, we'll return a JSON response with status code 200
    response_data = {
        'name': name[:-1],
        'sci_name': sci_name,
        'confidence': str(confidence),
    }

    return jsonify(response_data)


@app.route('/')
def index():
    return "Flask Server"


if __name__ == '__main__':
    app.run(host='192.168.56.1', port=5000, debug=True)
