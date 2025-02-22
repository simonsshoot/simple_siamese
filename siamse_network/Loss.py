import torch
import torch.nn.functional as F

'''继承 torch.nn.Module 使得这个类可以被 PyTorch 识别为一个神经网络模块，并且可以像其他 PyTorch 模块一样使用'''
class ContrastiveLoss(torch.nn.Module):
  def __init__(self,margin):
    '''调用了父类 torch.nn.Module 的构造函数，确保 ContrastiveLoss 类正确地继承了 torch.nn.Module 的所有属性和方法'''
    super(ContrastiveLoss,self).__init__()
    self.margin = margin  

  def forward(self,x1,x2,y):
    '''pairwise_distance计算两个向量的距离'''
    dist=F.pairwise_distance(x1,x2)
    '''loss最小:相似文本——y=0距离最小；pow表示幂运算，torch.clamp_min_ 函数用于确保 self.margin - dist 的值不小于 0，避免负数的出现，也就是说如果是不相似文本，他们的距离最小为0，且添加一个关于它的惩罚项'''
    total_loss=(1-y)*torch.pow(dist,2)+y*torch.pow(torch.clamp_min_(self.margin-dist,min=0.0),2)
    #所有样本对的loss取平均
    loss=torch.mean(total_loss)
    return loss


if __name__ == '__main__':
  x1=torch.randint(0,5,(4,3,3))
  x2=torch.randint(0,5,(4,3,3))
  y=torch.randint(0,2,(4,1))
  loss=ContrastiveLoss(margin=0.2)(x1,x2,y)
  print(loss)