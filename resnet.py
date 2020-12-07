import  torch
from    torch import  nn
from    torch.nn import functional as F


class Resblk(nn.Module):                                 #建立残差网络，
    '''
        残差块由两个3*3卷积，短接层由1*1的卷积层组成
    '''
    def __init__(self,channal_in,channal_out,stride = 1):
        super(Resblk,self).__init__()

        self.conv1 = nn.Conv2d(channal_in,channal_out,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(channal_out)
        self.conv2 = nn.Conv2d(channal_out,channal_out,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(channal_out)

        self.extra = nn.Sequential()            #  保证短接层输出通道数与卷积出的通道数一样           
        if channal_in != channal_out:                                       
            self.extra = nn.Sequential(
                            nn.Conv2d(channal_in,channal_out,kernel_size=1,stride=stride),
                            nn.BatchNorm2d(channal_out)
                                     )
    
        
    def forward(self,x):
        orginal_data = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.extra(orginal_data) + x
        return x

class Resnet_pic(nn.Module):                   #输入为200*200像素,传入要分类个数
    def __init__(self,num_class=5):
        super(Resnet_pic,self).__init__()
         #[b,64,w,h]>[b,64,w/2,h/2]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride = 2,padding = 0),
            nn.BatchNorm2d(64)
            )

        #follow 4 blocks 
        #[b,64,w/2,h/2]>[b,64,w/4,h/4]
        self.bk1= Resblk(64,128,stride=2)   
        #[b,128,w,h]
        self.bk2 = Resblk(128,256,stride=2)
        #[b,128,w/16,h/16]
        self.bk3 = Resblk(256,512,stride=2)
        #[b,128,w,h]
        


        self.fc_unit = nn.Sequential(
            nn.Linear(3*3*512,100),
            nn.ReLU(),
            nn.Linear(100,num_class)
        )
        '''
        x = torch.randn(5,3,200,200)
        x= self.conv1(x)     
        x = self.bk1(x)
        x = self.bk2(x)
        x = self.bk3(x)
        print(x.shape)
        '''

    def forward(self,x):
        x= self.conv1(x)

        x = F.relu(x)
        x = self.bk1(x)
        x = self.bk2(x)
        x = self.bk3(x)
        x = F.adaptive_avg_pool2d(x, [3, 3])
        x = x.view(x.size(0), -1)
        x = self.fc_unit(x)
        return x



def main():
    a= Resnet_pic(num_class=5)
    x = torch.randn(5,3,200,200)
    out = a(x)
    print('resnet:', out.shape)



if __name__ == "__main__":
    main()
        



        
