import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import os
from google.cloud import vision
from matplotlib.patches import Rectangle
import requests
from PIL import Image
from io import BytesIO
import streamlit as st


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



# Set your API key
api_key = "bD3oqNdx8FZyPeuv7UAuUZgV"

# Define the API endpoint
endpoint = "https://api.remove.bg/v1.0/removebg"

# Prepare the headers
headers = {
    "X-Api-Key": api_key,
}



def process_image(image_path):
    

    # Load the image
    image = cv2.imread(image_path)

        # Set your API key
    files = {  "image_file": open(image_path, "rb"),}

    # Call the API
    response = requests.post(endpoint, headers=headers, files=files)

    # Check if the response is successful
    if response.status_code == 200:
        # Read the output image
        output_image = Image.open(BytesIO(response.content))
        output_image = np.array(output_image)

        # Show the image using cv2_imshow
        cv2_imshow(output_image)
    else:
        print(f"Error: {response.content}")

    gray = cv2.cvtColor(output_image, cv2.COLOR_RGBA2GRAY) 
    edged = cv2.Canny(gray, 30, 200) 

    # Finding Contours 
    contours, hierarchy = cv2.findContours(edged, 
      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    # Print the number of contours found
    #print("Number of Contours found = " + str(len(contours))) 

    # List to store ROI images
    roi_images = []

    # Draw bounding rectangles and extract ROI
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = output_image[y:y+h, x:x+w]  # Extract ROI
        roi_images.append(roi)

    # Show extracted ROIs
    for i, roi in enumerate(roi_images):
        #cv2_imshow(roi)
        #cv2.imwrite(f"/content/roi_f.png", roi)  # Save the ROI
        break
    # Return the result
    return roi




def main():
    st.title("Streamlit App for Image Processing")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    uploaded_file = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    
    cv2.imwrite('uploaded_image.jpg',uploaded_file)
    if uploaded_file is not None:
        with open('uploaded_image.jpg', 'wb') as f:
            f.write(uploaded_file.getvalue())
        result_image = process_image('uploaded_image.jpg')

        alpha = 2.5
        beta = 1
        result_image = cv2.convertScaleAbs(result_image, alpha=alpha, beta=beta)
        st.image(result_image)
        client = vision.ImageAnnotatorClient()
        success, FdBack_ID = cv2.imencode('.jpg', result_image[int(result_image.shape[1]/7):,int(result_image.shape[0]/1.5):])

        image_ID = vision.Image(content=FdBack_ID.tobytes())
        response_ID = client.text_detection(image=image_ID)

        for text in response_ID.text_annotations[1:]:
            vertices = text.bounding_poly.vertices
            x = [vertex.x for vertex in vertices]
            y = [vertex.y for vertex in vertices]
            rect = Rectangle((x[0], y[0]), x[2] - x[0], y[2] - y[0], linewidth=1, edgecolor='r', facecolor='none')

        m = ''
        for x in response_ID.text_annotations[0].description.split('\n'):
            x = remove_non_english_arabic(x)
            m = m+x

        st.image(result_image[int(result_image.shape[1]/9):, int(result_image.shape[0]/1.5):], use_column_width=True)

        st.write("Extracted Text:")
        st.write(m)


if __name__ == "__main__":
    main()
