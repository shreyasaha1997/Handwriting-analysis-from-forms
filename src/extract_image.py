import cv2
import numpy as np

image = cv2.imread('Forms/F4.jpg')
# image = cv2.resize(image,(800,800))
cv2.imshow('main',image)
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
        if cv2.contourArea(ctr) < 1500:
            continue
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI
        roi = image[y:y+h, x:x+w]
        # cv2.imwrite('disintegrated blocks/image' + str(i) + '.png',roi)
        # cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
        # cv2.imwrite('bla' + str(i) + '.png',roi)
        images.append(roi)
    # cv2.imshow('lines',image)
    cv2.waitKey(0)
    return images

def word_seperation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (1000,100))
    image = cv2.resize(image, (1000,100))

    cv2.imshow('gr',gray)
    ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((1,1), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    # find contours
    im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    images = []

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        if cv2.contourArea(ctr) < 10:
            continue
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI
        roi = image[y:y + h, x:x + w]
        cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
        # cv2.imwrite('bla' + str(i) + '.png',roi)
        images.append(roi)
    cv2.imshow('lines', image)
    cv2.waitKey(0)
    return images


images = line_sep(image)
# cv2.imshow('lines',images)
for i in images:
    words = word_seperation(i)
cv2.waitKey(0)