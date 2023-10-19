
import streamlit as st
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import os
from google.cloud import vision
from matplotlib.patches import Rectangle
import matplotlib as plt

# Create a Vision API client
client = vision.ImageAnnotatorClient()

# Set the path to your service account key JSON file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'wise-baton-402315-a08c3e5df3fd.json'
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





def process_image(image_path):
    

    # Load the image
    image = cv2.imread(image_path)

    # Extract the bright object using GMM
    def extract_bright_object_gmm(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        flattened_hsv = hsv.reshape(-1, 3)
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0).fit(flattened_hsv)
        labels = gmm.predict(flattened_hsv).reshape(hsv.shape[:2])

        # Find the largest connected component
        _, labels, stats, _ = cv2.connectedComponentsWithStats(labels.astype(np.uint8), connectivity=8)
        largest_component = np.argmax(stats[1:, -1]) + 1
        mask = (labels == largest_component).astype(np.uint8)
        return cv2.bitwise_and(image, image, mask=mask)

    # Extract the bright object
    extracted_object = extract_bright_object_gmm(image)

    # Convert the extracted object to grayscale
    gray = cv2.cvtColor(extracted_object, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Return the result
    return image[y:y+h,x:x+w]




def main():
        st.title("Your OCR App")
	    
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")

        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
	
	    # Example usage
        result_image = image
	

	
        alpha = 1.5 # Contrast control (1.0-3.0)
        beta = 60 # Brightness control (0-100)
	
	
        result_image = cv2.convertScaleAbs(result_image, alpha=alpha, beta=beta)
	
	
	
	
        client = vision.ImageAnnotatorClient()
        success, FdBack_ID = cv2.imencode('.jpg', result_image [int(result_image.shape[1]/7): , int(result_image.shape[0]/1.5):])
	
        image_ID = vision.Image(content=FdBack_ID.tobytes())
        response_ID = client.text_detection(image=image_ID)
	
	# Extract and draw bounding boxes around text
        for text in response_ID.text_annotations[1:]:
                vertices = text.bounding_poly.vertices
                x = [vertex.x for vertex in vertices]
                y = [vertex.y for vertex in vertices]
                rect = Rectangle((x[0], y[0]), x[2] - x[0], y[2] - y[0], linewidth=1, edgecolor='r', facecolor='none')
	
	
        m = ''
        for x in response_ID.text_annotations[0].description.split('\n'):
                x= remove_non_english_arabic(x)
	
                m = m+x
	
                print(x)
	
        plt.imshow(result_image [int(result_image.shape[1]/9): , int(result_image.shape[0]/1.5):])
	
	
	
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
