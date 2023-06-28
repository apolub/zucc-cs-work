import cv2
from skimage import io

img = cv2.imread("../chromosome_classify/save/train/1/216w0078.043.K.R.PNG",0)
# blurred = cv2.GaussianBlur(img,(3,3),0)
# gaussImg = cv2.Canny(blurred, 10, 70)
gaussImg = cv2.Canny(img, 10, 70)
cv2.imshow("Img",gaussImg)
cv2.waitKey(0)
io.imsave('../chromosome_classify/save/train/1/316w0078.043.K.R.PNG',gaussImg)