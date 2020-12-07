'''
Author : Benjamin V
this file use for dealing the data 
'''

import torch
from torch import nn,optim
from resnet import Resnet_pic
from PIL import Image
import os,sys,glob
import random,csv
import numpy as np
import matplotlib.pyplot as plt
from   torch.nn import functional as F
from    torch.utils.data import Dataset, DataLoader
from    torchvision import transforms

'''
使用方法
db = face('D:/py/deeplearn/.vscode/xiangmu/rewrite_face/data',200,'train')    #数据夹path，resize大小，数据集模式
loader = DataLoader(db,batch_size=20,shuffle=True,num_workers=8)    #num_works  是进程数，

for epoch in loader:
    for i,(x,y) in eunmerate(laber):
'''


class face(Dataset):

    def __init__(self,root,resize,mode):
        super(face,self).__init__()

        self.root=root
        self.resize = resize                     #把数据存储下来

        self.namelabel = {}                      #类别以列表存下来

        list_picname = sorted(os.listdir(os.path.join(root)))
        for name in list_picname:
            if not os.path.isdir(os.path.join(root,name)):
                continue
            self.namelabel[name] = len(self.namelabel.keys())  
            #这里很巧妙把序号i换成keys值列表长度，也实现i功能，在添加列表可以借鉴

        
        self.images,self.labels = self.load_csv('images.csv')

        if mode=='train': # 60%
            self.images = self.images[:int(0.7*len(self.images))]
            self.labels = self.labels[:int(0.7*len(self.labels))]
        elif mode=='val': # 20% = 60%->80%
            self.images = self.images[int(0.7*len(self.images)):int(0.9*len(self.images))]
            self.labels = self.labels[int(0.7*len(self.labels)):int(0.9*len(self.labels))]
        else: # 20% = 80%->100%
            self.images = self.images[int(0.9*len(self.images)):]
            self.labels = self.labels[int(0.9*len(self.labels)):]
    

    def generate_csv(self,filename):               
        #把图片的路径，标签写入csv中
        if not os.path.exists(os.path.join(self.root,filename)):
            images = []
            for name in self.namelabel.keys():
                images  += glob.glob(os.path.join(self.root, name, '*.png'))    #读取name文件夹所有符合的文件
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            #print(len(images), images)
            random.shuffle(images)                             #打乱列表顺序，打散图片顺序

            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]               #取所对应文件夹名，即为图片类别，os.sep 返回分隔符
                    label = self.namelabel[name]
                    
                    writer.writerow([img, label])              #一次写入一行
                print('writen into csv file:', filename)

            
    '''
        处理思想，先将所有图片目录读进来，再打散，然后通过文件夹名字命名标签。
    '''

    def load_csv(self,filename):
        #读取csv文件
        if not os.path.exists(os.path.join(self.root,filename)):
            self.generate_csv(filename)

        images,label = [],[]

        with open(os.path.join(self.root,filename)) as f: 
            reader = csv.reader(f)
            for row in csv.reader(f):
                img,lab = row
                lab = int(lab)                #
                images.append(img)
                label.append(lab)

        assert len(images) == len(label)

        return images,label


    def __len__(self):
        return len(self.images)
    def denormalize(self, x_hat):                                  #为了可视化，打印出图片的状态
    
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x

    def __getitem__(self,idx):               #复写__fetitem__方法，把张图片数据进行处理，转化为tensor

        img,label = self.images[idx],self.labels[idx]

        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),                            #PLI库读图片的方法
            transforms.Resize((int(self.resize*1.2),int(self.resize*1.2))),   #这里*1.2是为了后面旋转后裁剪黑边要小
            transforms.RandomRotation(10),
            transforms.CenterCrop(self.resize),                               #这里统一为设置的图片大小
            transforms.ToTensor(),                                            #rgb数值压缩到0-1之间
            transforms.Normalize(mean=[0.485, 0.456, 0.406],                  #把数据整成 （0,1）的分布，0mean 1std 在-1--1之间
                                 std=[0.229, 0.224, 0.225])
            ])

        img = tf(img)

        label = torch.tensor(label)

        return img, label

#数据读入规则。把各个分类分别放到各个文件夹中，再把文件放在总文件中，给出总文件夹路径
#dataloader 会访问dataset类__getitem__方法，并按batch_size进行打包，返回值是由各个batch打包组成的列表,每个列表又由
#一个图片组成的列表，一个由标签组成的列表
def main():

    import  visdom
    import  time
    import  torchvision

    viz= visdom.Visdom()
    db = face('D:/py/deeplearn/.vscode/xiangmu/rewrite_face/data',200,'train')    #数据夹path，resize大小，数据集模式

    x,y = next(iter(db))

    viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))
    
    loader = DataLoader(db,batch_size=20,shuffle=True,num_workers=8)    #num_works  是进程数，

    for x,y in loader:
        viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
        viz.text(str(x.numpy()), win='label', opts=dict(title='x'))

        time.sleep(10)

if __name__ == '__main__':
    main()




        

        
        



         


