#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import matplotlib.pyplot as plt

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Load an image from file
image_path = "C:\\Users\\pooja\\Downloads\\merlin.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the original image
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")
plt.show()


# In[7]:


# Resize the image to match the model's expected sizing
image = cv2.resize(image, (224, 224))

# Expand the dimensions to match the model's expected input shape
input_image = np.expand_dims(image, axis=0)

# Preprocess the image for the model
input_image = preprocess_input(input_image)

# Make predictions
predictions = model.predict(input_image)

# Decode and print the top-3 predicted classes
decoded_predictions = decode_predictions(predictions, top=3)[0]
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")


# In[ ]:





# In[ ]:




