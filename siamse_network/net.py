import torch
import torch.nn.functional as F
import torch.nn as nn

class SiameseNetwork(nn.Module):
  def __init__(self):
    super(SiameseNetwork, self).__init__()

    self.cnn=nn.Sequential(
      #输出通道128：生成128个特征图，stride步幅
      nn.Conv2d(1, 128, kernel_size=5,stride=3,padding=2),
      #inplace=true：直接原地修改，不创建新的张量
      nn.ReLU(inplace=True),
      #局部响应归一化LRN:对输入的特征图进行归一化处理，以增强模型的泛化能力
      nn.LocalResponseNorm(5,alpha=0.001,beta=0.75,k=2),
      nn.MaxPool2d(kernel_size=3,stride=2),
      nn.Dropout2d(p=0.5),
    )
    self.fc=nn.Sequential(
      nn.Linear(2048,512),
      nn.ReLU(inplace=True),
      nn.Dropout2d(p=0.5),
      nn.Linear(512,128),
      nn.ReLU(inplace=True),
      nn.Linear(128,2)
    )
  '''forward_once处理单个输入数据'''
  def forward_once(self, x):
    output=self.cnn(x)
    '''经典cv操作！将图像展平到一维，-1自动计算维度，这样才能到全连接层！'''
    '''通道：图像中每个像素的颜色信息的分量！灰度图只有一个通道，彩色图有三个通道'''
    output=output.view(output.size()[0],-1)
    output=self.fc(output)
    return output
  
  def forward(self, input1, input2):
    output1=self.forward_once(input1)
    output2=self.forward_once(input2)
    return output1, output2
  
if __name__=='__main__':
  #测试
  '''生成随机图像张量，第一个1为批量，第二个1为通道，第三个28为高，第四个28为宽'''
  t=torch.randn((1,1,28,28))
  net=SiameseNetwork()
  output1,output2=net(t,t)
  print(output1,output2)