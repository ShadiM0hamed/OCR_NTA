import os
from google.cloud import vision

from matplotlib.patches import Rectangle


import streamlit as st

import cv2
import re
import pytesseract
from translate import Translator
import numpy as np


translator= Translator(from_lang="fa", to_lang="en") # Set the target language (in this case, French)

import os
from google.cloud import vision

# Set the path to your service account key JSON file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'wise-baton-402315-a08c3e5df3fd.json'



def id_borderer(image):
    # Load the pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    x, y, w, h = 0, 0, 0, 0  # Initialize x and y

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=5, minSize=(70, 70))
    try:
        for (x, y, w, h) in faces:
            # ...
            x1, y1 = x, y  # Top-left corner
            x2, y2 = x + int(w*4.5)+40, y + int(h*2.5)+40  # Bottom-right corner
    except:
        pass

    # Select the region using list slicing
    id_mask = image[y1:y2, x1:x2]

    return id_mask

def remove_non_english_arabic(text):
    # Define the regex pattern for English and Arabic characters

    # Join the matches back into a string
    cleaned_text = ' '
    cleaned_text = cleaned_text.replace(" ", "")
    if cleaned_text == '.':
      cleaned_text = '0'
    elif cleaned_text == '؛':
      cleaned_text = '4'
    elif cleaned_text == '،' or cleaned_text == ',':
      cleaned_text = '0'
    elif cleaned_text == ' ':
      pass
    cleaned_text = text
    return cleaned_text



# Create a Vision API client
client = vision.ImageAnnotatorClient()
def main():
    st.title("Your OCR App")
    
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Use uploaded_file like an open file in Python.
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        img = id_borderer(image)
        img = cv2.resize(img, (1080, 480))
        # Add other image processing steps here...

        # Perform OCR on the image
        # Add your OCR code here...

        # Display results in Streamlit
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        # Add other Streamlit components for displaying results...

	###############
        alpha = 1.7 # Contrast control (1.0-3.0)
        beta = 60 # Brightness control (0-100)

        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


        Id_mask = img[300:,410:]
        name_mask = img[10:310,410:]
	



        success, FdBack_ID = cv2.imencode('.jpg', Id_mask)
        success, FdBack_Name = cv2.imencode('.jpg', name_mask)


	# Perform OCR on the image
        image_ID = vision.Image(content=FdBack_ID.tobytes())
        response_ID = client.text_detection(image=image_ID)

	# Perform OCR on the image
        image_Name = vision.Image(content=FdBack_Name.tobytes())
        response_Name = client.text_detection(image=image_Name)

        col1, col2 = st.columns(2)
        col1.image(Id_mask, caption='ID Mask', use_column_width=True)
        col2.image(name_mask, caption='Name Mask', use_column_width=True)


	# Display the imagea

	# Extract and draw bounding boxes around text
        for text in response_ID.text_annotations[1:]:
                vertices = text.bounding_poly.vertices
                x = [vertex.x for vertex in vertices]
                y = [vertex.y for vertex in vertices]
                rect = Rectangle((x[0], y[0]), x[2] - x[0], y[2] - y[0], linewidth=1, edgecolor='r', facecolor='none')
	

        for text in response_Name.text_annotations[1:]:
                vertices = text.bounding_poly.vertices
                x = [vertex.x for vertex in vertices]
                y = [vertex.y for vertex in vertices]
                rect = Rectangle((x[0], y[0]), x[2] - x[0], y[2] - y[0], linewidth=1, edgecolor='r', facecolor='none')



        m = ''
        for x in response_ID.text_annotations[0].description.split('\n'):
                x= remove_non_english_arabic(x)
                m = m+x

        #m = translator.translate(m) # Text to be translated
        print(m)

        m2 = ''
        for x in response_Name.text_annotations[0].description.split('\n'):
                x= remove_non_english_arabic(x)
                m2 = m2+x

        #m2 = translator.translate(m) # Text to be translated
        print(m2)
        st.markdown(f"**Translated ID Text:** {m}")
        st.markdown("**Translated Name Text:** {}".format(response_Name.text_annotations[0].description.split('\n')))

if __name__ == "__main__":
    main()

