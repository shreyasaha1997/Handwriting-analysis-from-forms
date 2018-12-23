import cv2
import numpy as np


image = cv2.imread('p3.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (700,700))

(thresh, im_bw) = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
im_bw = cv2.resize(im_bw,(700,700))

rows = image.shape[0]
columns = image.shape[1]

cmin = columns + 100000
cmax = -1
rmin = rows + 100000
rmax = -1

for r in range(0,rows):
    for c in range(0,columns):
        if im_bw[r][c] == 255:
            if cmin > c:
                cmin = c
            if cmax < c:
                cmax = c
            if rmin > r:
                rmin = r
            if rmax < r:
                rmax = r

print(cmin)
print(cmax)
print(rmin)
print(rmax)



cv2.waitKey(0)
cv2.destroyAllWindows()