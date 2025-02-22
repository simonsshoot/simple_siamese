import pandas as pd
from PIL import Image
from pathlib import Path
import os
import torch

class Dataset:
  def __init__(self, data_dir,mode=None, transform=None):
    #将其处理为纯路径，path包提供多种处理不同操作系统的格式的方法
    self.data_dir = Path(data_dir)
    '''注意不同引用函数查看参数'''
    self.data_df=pd.read_csv(os.path.join(data_dir,f"{mode}.csv"),names=["imageA", "imageB", "label"])
    self.transform = transform
    self.mode=mode


  '''dataset类必须实现的方法'''
  def __len__(self):
    return len(self.data_df)
  
  def __getitem__(self, idx):
    imgA_path = str(self.data_dir/f"{self.data_df.at[idx, 'imageA']}")
    imgB_path = str(self.data_dir/f"{self.data_df.at[idx, 'imageB']}")
    #图像操作，L转为灰度图像
    imga = Image.open(imgA_path).convert("L")
    imgb = Image.open(imgB_path).convert("L")

    if self.transform:
      imga = self.transform(imga)
      imgb = self.transform(imgb)
    #提取指定idx行的标签
    return imga, imgb,torch.tensor([self.data_df.at[idx, 'label']],dtype=torch.float32)


if __name__ == '__main__':
  #测试
  data_dir="siamese/dataset"
  data=Dataset(data_dir,mode="train",transform=None)
  imga,imgb,label=data[0]
  print(imga.shape,imgb.shape,label)
