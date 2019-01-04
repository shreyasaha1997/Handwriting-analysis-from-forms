import cv2
import numpy as np
from Code1.word_detection import wordSegmentation, prepareImg

image = cv2.imread('Forms/p1.png')
im1 = cv2.imread('Forms/p1.png')
image = cv2.resize(image, (1000,1000))
im1 = cv2.resize(image, (1000,1000))

#grayscale

def line_sep(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #binary
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

    #dilation
    kernel = np.ones((1,50), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
        #find contours
    im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    images = []

    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        if cv2.contourArea(ctr) < 3000:
            continue
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI
        roi = image[y:y+h, x:x+w]
        cv2.rectangle(im1,(x,y),( x + w, y + h ),(90,0,255),2)
        # cv2.imwrite('bla' + str(i) + '.png',roi)
        images.append(roi)
    return images

def word_seperation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # binary
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # dilation
    kernel = np.ones((10, 25), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    # find contours
    im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    images = []

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        if cv2.contourArea(ctr) < 1000:
            continue
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI
        roi = image[y:y + h, x:x + w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2)
        images.append(roi)
    return (image, images)


lines = line_sep(image)
cv2.imshow('marked areas',im1)

for line in lines:
    word, individual_words = word_seperation(line)
    # cv2.imshow('line',word)
    for (i,w) in enumerate(individual_words):
        cv2.imshow('word',w)
        cv2.imwrite('bla' + str(i) + '.png',w)
        cv2.waitKey(0)
    cv2.waitKey(0)


cv2.waitKey(0)