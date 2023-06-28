from PyQt5 import QtWidgets, QtCore, QtWidgets,QtGui
from PyQt5.QtWidgets import QFileDialog,QColorDialog,QAction
import sys
import  Ui_test
import sys,os
import cv2
import numpy as np

class PicPairing(Ui_test.Ui_MainWindow):
    def __init__(self):
        self.pic_pair_window=QtWidgets.QMainWindow()#生成一个qmainwindow
        super().setupUi(self.pic_pair_window)#调用父类的setupUI函数
        self.pushButton.clicked.connect(self.selec_img_dir)
        self.pushButton_2.clicked.connect(self.pairing_operation)
       
      
        
        
    def selec_img_dir(self):
        home_dir = os.getcwd()#获取初始路径，初始路径默认为当前路径
        self.dir = QFileDialog.getExistingDirectory(self.pic_pair_window, 'Open file', home_dir)#获取文件路径
        files=os.listdir(self.dir)
        self.img_list=[]
        for file in files:
            filename=os.path.join(self.dir,file)
            img=cv2.imread(filename,-1)
            img=cv2.resize(img,(60,80))
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            self.img_list.append(img)
        row,col,depth=self.img_list[0].shape
        new_row=6
        new_col=8
        merged_img=np.zeros((new_row*row,new_col*col,3),np.uint8)
        merged_img[:,:,:]=255
        for i in range(new_row):
            for j in range(new_col):
                if (i*new_col+j)<len(files):
                    merged_img[row*i:row*(i+1),col*j:col*(j+1)]=self.img_list[i*new_col+j]
        y,x,depth=merged_img.shape
        frame=QtGui.QImage(merged_img,x,y,QtGui.QImage.Format_RGB888)
        pixmap=QtGui.QPixmap.fromImage(frame)
        self.label.setPixmap(pixmap)

                
       

    def pairing_operation(self):
        merged_img_2_list=[]
        for index in range(int(len(self.img_list)/2)):
            img1=self.img_list[index*2]
            idx1 = np.argwhere(np.all(img1[:,:] == 255, axis=0))
            img1=np.delete(img1, idx1, axis=1)
            img2=self.img_list[index*2+1]
            idx2 = np.argwhere(np.all(img2[:,:] == 255, axis=0))
            img2=np.delete(img2, idx2, axis=1)
            new_img=np.hstack((img1,img2))
            new_img=cv2.resize(new_img,(80,80))
            merged_img_2=np.zeros((120,80,3),np.uint8)
            merged_img_2[:,:,:]=255
            merged_img_2[0:80,0:80]=new_img
            cv2.putText(merged_img_2,'________',(20,90),cv2.FONT_HERSHEY_PLAIN,1,(123,224,100),2)
            cv2.putText(merged_img_2,str(index),(30,115),cv2.FONT_HERSHEY_PLAIN,2,(123,224,100),2)
            merged_img_2_list.append(merged_img_2)
        row,col,depth=merged_img_2_list[0].shape
        new_row=4
        new_col=6
        merged_img=np.zeros((new_row*row,new_col*col,3),np.uint8)
        merged_img[:,:,:]=255
        for i in range(new_row):
            for j in range(new_col):
                if (i*new_col+j)<len(merged_img_2_list):
                    merged_img[row*i:row*(i+1),col*j:col*(j+1)]=merged_img_2_list[i*new_col+j]
        y,x,depth=merged_img.shape
        frame=QtGui.QImage(merged_img,x,y,QtGui.QImage.Format_RGB888)
        pixmap=QtGui.QPixmap.fromImage(frame)
        self.label_2.setPixmap(pixmap)
        
    
   

if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    app = QtWidgets.QApplication(sys.argv)
    pic_pairing_ui = PicPairing()#class实例
    pic_pairing_ui.pic_pair_window.show()#显示窗口
    sys.exit(app.exec_())