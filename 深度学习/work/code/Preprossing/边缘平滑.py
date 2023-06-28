import numpy as np 
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
from skimage import io


def padding(image,ksize):

    h = image.shape[0]
    w = image.shape[1]
    c = image.shape[2]

    pad = ksize // 2 

    out_p = np.zeros((h+2*pad,w+2*pad,c))

    out_copy = image.copy()

    out_p[pad:pad+h,pad:pad+w,0:c] = out_copy.astype(np.uint8)

    return out_p


def gaussian(image,ksize,sigma):

    """
    1. padding
    2. 定义高斯滤波公式与卷积核
    3. 卷积过程

    高斯卷积卷积核是按照二维高斯分布规律产生的，公式为：

    G(x,y) = (1/(2*pi*(sigma)^2))*e^(-(x^2+y^2)/2*sigma^2)

    唯一的未知量是sigma，在未指定sigma的前提下，可以通过下列参考公式让程序自动选择合适的
    sigma值：

    sigma =  0.3 *((ksize-1)*0.5-1) + 0.8

    @ 如果mode为default，则返回abs值，否则返回unit8值
    """
        

    pad = ksize//2

    out_p = padding(image,ksize) # padding之后的图像
    # print(out_p)

    h = image.shape[0]
    w = image.shape[1]
    c = image.shape[2]

    # 高斯卷积核

    kernel = np.zeros((ksize,ksize))
    for x in range(-pad,-pad+ksize):
        for y in range(-pad,-pad+ksize):
            kernel[y+pad,x+pad] = np.exp(-(x**2+y**2)/(2*(sigma**2)))
    kernel /= (sigma*np.sqrt(2*np.pi))
    kernel /=  kernel.sum()

    # print(kernel)

    tmp = out_p.copy()

    # print(tmp)

    for y in range(h):
        for x in range(w):
            for z in range(c):

                out_p[pad+y,pad+x,z] = np.sum(kernel*tmp[y:y+ksize,x:x+ksize,z])


    out = out_p[pad:pad+h,pad:pad+w].astype(np.uint8)
    # print(out)

    return out
	
if __name__ == "__main__":

    path = '../chromosome_classify/save/train/1/116w0078.043.K.R.PNG'

    img = cv.imread(path)
    gaussian_img = gaussian(img,3,0.8)
    # cv.imshow('Original Image',img)
    # cv.waitKey()
    cv.imshow('Gaussian Image',gaussian_img)
    cv.waitKey()
    io.imsave('../chromosome_classify/save/train/1/216w0078.043.K.R.PNG',gaussian_img)

     # 高斯卷积核

    kernel = np.zeros((ksize,ksize)) # 创建一个卷积核大小的全0二维矩阵
    for x in range(-pad,-pad+ksize):
        for y in range(-pad,-pad+ksize):
            kernel[y+pad,x+pad] = np.exp(-(x**2+y**2)/(2*(sigma**2))) # 给卷积核内赋值
    kernel /= (sigma*np.sqrt(2*np.pi)) # 计算平均值
    kernel /=  kernel.sum() # 加总
