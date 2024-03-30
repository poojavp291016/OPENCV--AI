#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load a sample image for demonstration
sample_image = cv2.imread('C:\\Users\\pooja\\Downloads\\image.jpg')

# Check if the image was loaded successfully
if sample_image is None:
    print("Error: Unable to load image.")
    exit()

# Resize the width of the image
new_width = 400  # Adjust the width as needed
aspect_ratio = sample_image.shape[1] / sample_image.shape[0]
new_height = int(new_width / aspect_ratio)
resized_image = cv2.resize(sample_image, (new_width, new_height))

# Convert the resized image to grayscale
gray_sample_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Detect faces in the resized image
faces = face_cascade.detectMultiScale(gray_sample_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(resized_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the resized image with detected faces
cv2.imshow('Facial Recognition', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




