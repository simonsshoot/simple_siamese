import os
import random
import glob
import shutil
#后两个包都是用于文件和目录操作

def sample_data():
  dp="data/dataset/data/train/"
  data_list=range(10000)
  print(data_list.head(5))
  with open("data/dataset/data/train_labels.txt") as f:
    labels=f.read().rstrip().split()

  img_path="dataset/images/"
  if not os.path.exists(img_path):
    os.makedirs(img_path)
  #data_images：0-10000的图片编号和对应的标签
  data_images=list(zip(data_list,labels))
  random.shuffle(data_images)
  datas=random.sample(data_images,1000)

  for i,(data,label) in enumerate(datas):
    #文件复制
    shutil.copy(os.path.join(dp,f"{data}.png"),os.path.join(img_path,f"{label}_{data}.png"))

def get_data_pair():
  datas=os.listdir("dataset/images/")
  #分割数据集为训练集和测试集
  random.shuffle(datas)
  partion=int(len(datas)*0.9)
  train_data=datas[:partion]
  test_data=datas[partion:]
  with open("dataset/test.csv","w")as f:
    for i in range(800):
      '''孪生网络，对比学习——样本对'''
      pairs=random.sample(test_data,2)
      #样本对标签
      cat=0 if pairs[0][0]==pairs[1][0] else 1
      f.write(f"images/{pairs[0]},images/{pairs[1]},{cat}\n")
  with open("dataset/train.csv","w")as f:
    for i in range(8000):
      pairs=random.sample(train_data,2)
      cat=1 if pairs[0][0]==pairs[1][0] else 0
      f.write(f"images/{pairs[0]},images/{pairs[1]},{cat}\n")
  

if __name__=="__main__":
  get_data_pair()
