import cv2
import numpy as np

image = cv2.imread('p2.png')

def filter_region(image, vertices):
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2])
    return cv2.bitwise_and(image, mask)

rows, cols = image.shape[:2]
r1 = 0
r2 = 0.1
while r2 <= 1:
    bottom_left = [cols * 0, rows * r2]
    top_left = [cols * 0, rows * r1]
    bottom_right = [cols * 1, rows * r2]
    top_right = [cols * 1, rows * r1]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    img = filter_region(image, vertices)
    cv2.imshow('bb',img)
    cv2.waitKey(0)
    r1 = r1 + 0.1
    r2 = r2 + 0.1
cv2.destroyAllWindows()