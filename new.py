import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
main_image = cv.imread('images/affected/zxzx1.jpg',0)

expected = 31500
size = (620,500)
image = cv.resize(main_image,size)

cv.imshow("grayscale",image)
img = cv.medianBlur(image,5)
cv.imshow("blur",img)

th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)

pix = np.sum(th2 == 0)
print(f"No. of pixels {pix}")
cv.imshow("output",th2)

if pix >= expected:
    print("Hypertensive retinopathy True")
else:
    print("Hypertensive retinopathy False")

cv.waitKey(0)