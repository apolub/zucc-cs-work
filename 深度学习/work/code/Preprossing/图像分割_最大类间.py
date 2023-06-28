import numpy as np
import cv2 as cv
from skimage import io
#该函数返回的第一个值就是输入的thresh值，第二个就是处理后的图像
img = cv.imread('../chromosome_classify/classification_24category/train/1/16w0078.043.K.R.PNG',0)
retVal, a_img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
# print("使用opencv函数的方法：" + str(retVal))
# io.imsave('../chromosome_classify/save/train/1/16w0078.043.K.R.PNG',a_img)
# cv.imshow("a_img",a_img)
# cv.waitKey()

# 图像腐蚀操作
# 首先设置卷积核大小，3x3，值置1，格式为numpy中的整数uint8，0到255.
# kernel = np.ones((5,5),np.uint8) 

# kernel = [[0,1,0],[1,1,1],[0,1,0]]
# kernel=np.array(kernel,np.uint8)
# 腐蚀操作，可以设置迭代次数，改动iterations看结果，0为不腐蚀
# erosion = cv.erode(a_img,kernel,iterations = 2)

# 开操作
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
erosion = cv.morphologyEx(a_img, cv.MORPH_OPEN, kernel)

# res = np.hstack((erosion))
cv.imshow('erosion', erosion)
cv.waitKey(0)
io.imsave('../chromosome_classify/save/train/1/116w0078.043.K.R.PNG',erosion)