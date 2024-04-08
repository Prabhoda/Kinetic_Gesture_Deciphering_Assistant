import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import numpy as np
from googletrans import Translator

# Load the pre-trained YOLO model
model = YOLO('best.pt')

# Read the COCO class list from a file
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Translation function
def translate_text(text, destination_language):
    translator = Translator()
    translation = translator.translate(text, dest=destination_language)
    return translation.text

# Streamlit app
st.title("Kinetic Gesture Deciphering Assistant")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif"], accept_multiple_files=False)

# Dropdown for language selection
language_options = ['Telugu',
                    'Hindi',
                    'English',
                    'Tamil',
                    'Kannada',
                    'Marathi',
                    'Malayalam',
                    'Urdu',
                    'Panjabi',
                    'Oriya',
                    'Konkani',
                    'Manipuri']
selected_language = st.selectbox("Select Language for Translation:", language_options)

if uploaded_file is not None:
    # Read the image
    frame = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Resize the frame to a fixed size (if necessary)
    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection on the frame
    results = model.predict(frame)
    detections = results[0].boxes.data
    px = pd.DataFrame(detections).astype("float")

    # Display the detected objects
    st.image(frame, channels="BGR", caption="Uploaded Image with Object Detection", use_column_width=True)

    detected_objects = []

    # Display the detected objects and their class labels
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        c = class_list[d]

        # Draw bounding boxes and class labels on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

        detected_objects.append(c)

    # Display the frame with objects detected
    st.image(frame, channels="BGR", caption="Detected Objects", use_column_width=True)

    # Translate detected objects
    translated_objects = [translate_text(obj, selected_language.lower()) for obj in detected_objects]
    st.write('Detected Objects: ', ", ".join(translated_objects))
