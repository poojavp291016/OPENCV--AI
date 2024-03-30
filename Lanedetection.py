#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[8]:


pip install docopt


# In[9]:


pip install moviepy


# In[10]:


pip install ipython


# In[11]:


pip install opencv-python


# In[ ]:





# In[14]:


import cv2


# In[15]:


import numpy as np


# In[16]:


class LaneDetector:
    def __init__(self):
        pass

    def detect_lanes(self, img):
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Define region of interest (ROI)
        mask = np.zeros_like(edges)
        height, width = img.shape[:2]
        vertices = np.array([[(100, height), (width // 2 - 50, height // 2 + 60),
                              (width // 2 + 50, height // 2 + 60), (width - 100, height)]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, np.array([]), minLineLength=50, maxLineGap=100)

        # Draw lines on original image
        line_img = np.zeros_like(img)
        self.draw_lines(line_img, lines)

        # Combine original image with detected lines
        result = cv2.addWeighted(img, 0.8, line_img, 1, 0)

        return result

    def draw_lines(self, img, lines, color=(255, 0, 0), thickness=2):
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def main():
    # Initialize LaneDetector
    detector = LaneDetector()

    # Read input video
    cap = cv2.VideoCapture('challenge_video.mp4')

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Detect lanes in the frame
            result = detector.detect_lanes(frame)

            # Display the result
            cv2.imshow('Lane Detection', result)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release video capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# In[ ]:




