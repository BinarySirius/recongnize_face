'''
Author : Benjamin V
this file use for predicting whos face 
'''

import torch
from torch import nn,optim
from resnet import Resnet_pic
from   torch.nn import functional as F
from train_byres import oneimg_tran
import cv2 


classfier = cv2.CascadeClassifier("D:/py/deeplearn/data/haarcascade_frontalface_default.xml")   #分类文件位置
color = (0, 255, 0)

model = Resnet_pic()
model.load_state_dict(torch.load('D:/py/deeplearn/.vscode/xiangmu/res_params.pkl'))                 #读取训练好的模型



cap = cv2.VideoCapture(0)                          #0内置，1外置
cap.set(cv2.CAP_PROP_FRAME_WIDTH,720)             #设置窗口大小,窗口大小默认是640*360
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)

while True:
    ret,frame = cap.read()                        #ret 是布尔值有返回为true,frame 为一帧图像
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #转换为灰度图片
    faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.3, minNeighbors = 1, minSize = (160, 160))
    #scale 为图片缩放比例，minNeighbors为最少检测几次为真，minsize 最小人脸尺寸
    if len(faceRects) > 0:            #大于0则检测到人脸                                   
            for faceRect in faceRects:  #单独框出每一张人脸
                x, y, w, h = faceRect        
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 3)  #方框坐标，大小，及宽度
                
                img = frame[y-5:y+h+5,x-5:x+w+5]
                img = cv2.resize(img,(200,200))            
                tensor_img =oneimg_tran(img)
                out = model(tensor_img)
                out = torch.max(out,1)[1].item()
                if out == 1:
                    name = 'tianyong huang'
                elif out ==3:
                    name = 'yunfeng li'
                elif out == 0:
                    name = 'jincong huang'
                elif out == 2:
                    name = 'jia'
                cv2.putText(frame,'%s'%name, 
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
                                2)                                     #字的线宽
                
                
                



    cv2.imshow("video",frame)
    
    if cv2.waitKey(200)== ord('q'):    #ord返回ascil值，waitkey()等待时间 类似实现帧率效果  
        break

cap.release()
cv2.destroyAllWindows()


