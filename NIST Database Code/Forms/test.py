import cv2
import numpy as np

from pdf2image import convert_from_path

for i in range(1,8):

	pdf_file = 'f' + str(i) + '.pdf'
	pages = convert_from_path(pdf_file, 500)
	imname = 'im'+str(i)+'.jpg'
	for page in pages:
	    page.save(imname, 'JPEG')



