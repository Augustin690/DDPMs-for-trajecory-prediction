import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_data(data):

    # 假设有100人，每人两列数据，总共200列
    num_pedestrians = 100
    num_steps = len(data)
    # 分别为每个行人创建x和y坐标的列表
    x = [[] for _ in range(num_pedestrians)]
    y = [[] for _ in range(num_pedestrians)]

    for i in range(num_steps):
        for j in range(num_pedestrians):
            x[j].append(data[i][2*j])    # 2*j是因为每个行人有两列数据，x坐标
            y[j].append(data[i][2*j+1])  # 2*j+1是y坐标

    # 绘制所有行人的轨迹
    img = mpimg.imread('MI.png')
    plt.figure(figsize=(10, 8))
    for j in range(num_pedestrians):
        plt.scatter(x[j], y[j], s=1.5)
    plt.imshow(img, extent=[0, 100, 0, 100])
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectories of 100 Pedestrians')
    plt.show()

    '''
    x[j], y[j] 代表第j个人的150个位置
    '''

data = np.loadtxt('pedestrians_positions_MI.csv')
print(data.shape)  # data.shape: 步数（可以是1）x（人数*2）
plot_data(data)