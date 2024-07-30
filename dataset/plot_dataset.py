import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_data(data):

    # Assuming 100 pedestrians and two columns of data per pedestrian, for a total of 200 columns
    num_pedestrians = 100
    num_steps = len(data)
    # Create separate lists of x and y coordinates for each pedestrian
    x = [[] for _ in range(num_pedestrians)]
    y = [[] for _ in range(num_pedestrians)]

    for i in range(num_steps):
        for j in range(num_pedestrians):
            x[j].append(data[i][2*j])    # 2*j because there are two columns of data for each pedestrian, the x-coordinate
            y[j].append(data[i][2*j+1])  # 2*j+1 y coordinate

    # Plotting the trajectory of all pedestrians
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
print(data.shape)  # data.shape: Number of steps (can be 1) x (number of pedestrians * 2)
plot_data(data)