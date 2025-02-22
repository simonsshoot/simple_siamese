import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from Dataset import Dataset
from Loss import ContrastiveLoss
from net import SiameseNetwork
from utils import imshow,show_plot
import matplotlib
import matplotlib.pyplot as plt


def visualize_example_data(dataset):
   # Viewing the sample of images and to check whether its loading properly
  vis_dataloader = DataLoader(dataset, shuffle=True, batch_size=8)
  dataiter = iter(vis_dataloader)
  example_batch = next(dataiter)
  print(example_batch[0].shape)
  concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
  imshow(torchvision.utils.make_grid(concatenated))
  print(example_batch[2].numpy())

def load_config():
  with open("config.yaml")as f:
    cfg=yaml.safe_load(f)
  return cfg

def train(model,opts,transform):
  dataset=Dataset(opts['data_dir'],opts['mode'],transform=transform)
  dataloader=DataLoader(dataset,batch_size=opts['batch'],shuffle=True,num_workers=4)

  loss=[]
  counter=[]
  iter_number=0
  '''对比损失函数，通常用于成对的样本输入，主要用于训练模型学习样本之间的相似性或距离。'''
  criterion = ContrastiveLoss(0.4)
  '''weight_decay，防止模型过拟合，惩罚项: 在使用 weight_decay 时，优化器在更新权重时，会将当前权重乘以一个小于1的值，通常是 1 - lr * weight_decay（其中 lr 是学习率）。这样做的目的是为了减少权重的大小，从而降低模型复杂度，增强模型的泛化能力'''
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=5e-3)

  for epoch in tqdm(range(opts['epoches']), desc="Training Progress"):
    #设置为 0 时表示数据加载在主进程中进行
    for i,data in enumerate(dataloader,0):
      #两个样本组成样本对+标签
      img1,img2,label=data
      '''每个训练批次前调用zero_grad清理梯度，防止梯度累积,这几步都是pytorch的范式'''
      optimizer.zero_grad()
      y1,y2=model(img1,img2)
      cur_loss=criterion(y1,y2,label)
      cur_loss.backward()
      #优化器更新参数
      optimizer.step()
      # if i%opts['print_freq']==0:
      #   print(f"{i}th iteration,loss:{cur_loss.item()}")
      # print(f"Epoch {epoch} finished,loss:{cur_loss.item()}")
      # iter_number+=len(dataloader)/opts['batch']

      counter.append(iter_number)
      loss.append(cur_loss.item())
      '''保存模型参数'''
      if epoch%opts['save_freq']==0:
        torch.save(model.state_dict(),f"model/model_{epoch}.pt")

  matplotlib.use('Agg')  # 使用非交互式后端，解决远程服务器可能无法显示图像的问题
  show_plot(counter,loss)
  plt.savefig('output_pics/loss.png') 
  return model


def test(model,opts,transform):
  test_dataset=Dataset(opts['data_dir'],mode='test',transform=transform)
  test_dataloader=DataLoader(test_dataset)

  model.eval()
  for i,data in enumerate(test_dataloader):
    x1,x2,label=data
    concat=torch.cat((x1,x2),dim=0)
    y1,y2=model(x1,x2)
    dist=F.pairwise_distance(y1,y2)
    if label == torch.FloatTensor([[1]]):
      label = "Different Signature"
    else:
      label = "Same Signature"

      '''可视化的常用函数，make_grid将多个图片拼接成一张图片可视化批次图像数据'''
    matplotlib.use('Agg')
    imshow(torchvision.utils.make_grid(concat))
    plt.savefig(f'output_pics/concat{i}.png')
    print(f"Predicted Distance:\t{dist.item()}")
    print(f"Actual Label:\t{label}")
    if i > 100:
      break
  

def main(opts):
  #数据预处理，图像转换为tensor
  transform=torchvision.transforms.Compose([
    transforms.ToTensor()
  ])

  net=SiameseNetwork()
  model=train(net,opts,transform)
  test(model,opts,transform)
  device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


if   __name__ == '__main__':
  opts=load_config()
  main(opts)