from deslant import deslant_image
import cv2
import numpy as np

# read and convert image to black and white
filename = "data/a02-000-s01-00.png"
out = "outputs/"
img = cv2.imread(filename)
p = deslant_image.RotateAndDeslantImage()
gray_image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
_, im = cv2.threshold(gray_image, 160, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# call deslant method
im = p.deslant_image(im)

# show images
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow("image", im)
cv2.namedWindow('image1',cv2.WINDOW_NORMAL)
cv2.imshow("image1", img)
cv2.waitKey(0)

# write output
cv2.imwrite(out + filename.split("/")[1] , im*255)
