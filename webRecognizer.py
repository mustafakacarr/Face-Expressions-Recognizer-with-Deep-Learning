# Load the pre-trained model
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained model
model = load_model('face-expression-model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Function to preprocess the image
def preprocess_image(img):
    img = img / 255.0
    img = cv2.resize(img, (48, 48))
    img = img.reshape(-1, 48, 48, 1)
    return img

# Function to detect and classify facial expressions
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        facec = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
           fc = gray_fr[y:y+h, x:x+w]

           roi = cv2.resize(fc, (48, 48))
           preprocessed_face = preprocess_image(roi)

           # Predict the facial expression using the loaded model
           prediction = model.predict(preprocessed_face)
           predicted_class = np.argmax(prediction)
           expression = "Angry" if predicted_class == 0 else "Sad"

           # Get the confidence score (probability) for the predicted class
           confidence_score = np.max(prediction) * 100

           # Draw a rectangle around the detected face
           cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)

           # Display the predicted expression and confidence score near the rectangle
           text = f"{expression} ({confidence_score:.2f}%)"
           cv2.putText(fr, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return fr

# Title of the application
st.title('Facial Expression Detection')

# Create a VideoCamera instance
camera = VideoCamera()

# Display the video stream and predicted facial expressions
if st.button('Start Detection', key=0):
    while True:
        frame = camera.get_frame()
        st.image(frame, channels='BGR', use_column_width=True)

        # To exit the loop and stop the video stream, click the 'Stop Detection' button
        if st.button('Stop Detection', key=1):
            break 

# Release the camera
camera.__del__()
