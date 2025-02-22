import numpy as np
import matplotlib.pyplot as plt


def imshow(img, text=None, should_save=False):
    #图像数据通常以numpy张量数组的形式存在
    npimg = img.numpy()
    #关闭坐标轴显示图像
    plt.axis("off")
    #如果要显示文本的话
    if text:
        plt.text(
            75,
            8,
            text,
            style="italic",
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 10},
        )
    '''使用plt.imshow显示图像，并调用plt.show()来呈现结果。注意这里使用了np.transpose将numpy数组的维度从(C, H, W)（通道数、高度、宽度）转换为(H, W, C)，这是因为matplotlib的imshow函数期望输入的图像数组维度是(H, W, C)或者(H, W)（灰度图像）'''
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#显示损失函数随迭代次数变化的图表
def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()