import cv2
import numpy as np

image = cv2.imread('Sentences/p3.png')
im1 = cv2.imread('Sentences/p3.png')
# image = cv2.resize(image, (700,700))
# im1 = cv2.resize(image, (700,700))

def word_seperate(image):
    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #binary
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

    #dilation
    kernel = np.ones((5,20), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    cv2.imshow('dilated',img_dilation)

    #find contours
    im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    images = []

    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI
        roi = image[y:y+h, x:x+w]
        cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
        xxx = im1[y:y+h, x:x+w]
        images.append(xxx)
    return image


