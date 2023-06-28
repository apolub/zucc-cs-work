# 导入库
import matplotlib
import matplotlib.pyplot as plt
import skimage.io as io
 
# # 将图像转为灰度图像
# from PIL import Image
# img = Image.open("../chromosome_classify/save/train/1/16w0078.043.K.R.PNG").convert('L')
# img.save('E://straight//3 greyscale.png')
 
# 读取灰度图像
img = io.imread("../chromosome_classify/save/train/1/116w0078.043.K.R.PNG")
 
# 对图像进行预处理，二值化
from skimage import filters
from skimage.morphology import disk
# 中值滤波
Img_Original = filters.median(img,disk(5))
# 二值化
BW_Original = Img_Original < 235

# 定义像素点周围的8邻域
#                P9 P2 P3
#                P8 P1 P4
#                P7 P6 P5
 
def neighbours(x,y,image):
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y],img[x_1][y1],img[x][y1],img[x1][y1],         # P2,P3,P4,P5
            img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9
 
# 计算邻域像素从0变化到1的次数
def transitions(neighbours):
    n = neighbours + neighbours[0:1]      # P2,P3,...,P8,P9,P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3),(P3,P4),...,(P8,P9),(P9,P2)
 
# Zhang-Suen 细化算法
def zhangSuen(image):
    Image_Thinned = image.copy() 
    changing1 = changing2 = 1
    while changing1 or changing2:  
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and  
                    2 <= sum(n) <= 6   and    # a: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # b: S(P1)=1  
                    P2 * P4 * P6 == 0  and    # c   
                    P4 * P6 * P8 == 0):         # d
                    changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and    
                    2 <= sum(n) <= 6  and       # a
                    transitions(n) == 1 and      # b
                    P2 * P4 * P8 == 0 and       # c
                    P2 * P6 * P8 == 0):            # d
                    changing2.append((x,y))    
        for x, y in changing2: 
            Image_Thinned[x][y] = 0
    return Image_Thinned

# 对染色体图像应用Zhang-Suen细化算法
BW_Skeleton = zhangSuen(BW_Original)
import numpy as np
BW_Skeleton = np.invert(BW_Skeleton)

# 显示细化结果
# fig, ax = plt.subplots(1, 2)
# ax1, ax2 = ax.ravel()
# ax1.imshow(img, cmap=plt.cm.gray)
# ax1.set_title('Original binary image')
# ax1.axis('off')
BW_Skeleton = np.abs(BW_Skeleton-255)
plt.imshow(BW_Skeleton, cmap=plt.cm.gray)
plt.axis('off')
plt.show()
io.imsave('../chromosome_classify/save2/train/1/16w0078.043.K.R.PNG',BW_Skeleton)
