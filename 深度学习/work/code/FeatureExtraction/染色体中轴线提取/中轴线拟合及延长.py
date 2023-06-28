# 利用opencv在染色体图像上画出中轴线
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
 
img = cv.imread('../chromosome_classify/save/train/1/116w0078.043.K.R.PNG')
binaryImg = cv.Canny(img,100,200)
h = cv.findContours(binaryImg, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
contours = h[0]
 
Org_img =  cv.imread('../chromosome_classify/save2/train/1/16w0078.043.K.R.PNG')
cv.drawContours(Org_img, contours, -1, (255,255,255), 1)
 
plt.axis('off')
plt.imshow(Org_img, cmap=plt.cm.gray)
plt.show()
 
cv.imwrite('../chromosome_classify/save/train/1/416w0078.043.K.R.PNG',Org_img)
# img = cv.imread('../chromosome_classify/save/train/1/116w0078.043.K.R.PNG')

# Org_img =  cv.imread('../chromosome_classify/save2/train/1/16w0078.043.K.R.PNG')