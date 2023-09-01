import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('c:\Coding\Python\Image Processing\ImageDetection\\ball_detection_model.h5')

# Load the ball detection labels
labels = ['ball', 'box']

# Function to preprocess the input image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to perform ball detection
def detect_ball(image_path):
    image = cv2.imread(image_path)
    processed_image = preprocess_image(image)

    # Make predictions using the trained model
    predictions = model.predict(processed_image)
    predicted_label = labels[np.argmax(predictions)]

    # Display the result
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if predicted_label == 'ball':
        print("The image contains a ball.")
    else:
        print("The image contain a box.")

# Provide the path to the image you want to test
image_path = 'c:\Coding\Python\Image Processing\ImageDetection\datatest\datatest4.jpg'
detect_ball(image_path)