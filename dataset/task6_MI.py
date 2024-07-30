import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

def update_positions(positions, waypoints_groups, current_indices, noise_scale=0.01):
    new_positions = np.copy(positions)
    for i in range(len(positions)):
        group_id = group_indices[i]
        waypoints = waypoints_groups[group_id]
        current_index = current_indices[i]
        if current_index < len(waypoints):
            goal = waypoints[current_index]
            if np.linalg.norm(positions[i] - goal) < 0.5:
                current_indices[i] += 1
                if current_indices[i] >= len(waypoints):
                    continue

            direction = goal - positions[i]
            norm = np.linalg.norm(direction)
            step_size = 0.003  # 设置新的步长
            direction = (direction / norm if norm != 0 else direction) * step_size
            noise = noise_scale * np.random.normal(size=positions[i].shape)
            new_positions[i] = positions[i] + direction + noise


    return new_positions, current_indices


def count_completed_pedestrians(current_indices, waypoints_groups):
    completed_count = 0
    for i, index in enumerate(current_indices):
        group_id = group_indices[i]
        if index >= len(waypoints_groups[group_id]):
            completed_count += 1
    return completed_count
'''
可以修改的部分↓
'''
# 定义不同的起始点和途经点
start_points = [(5, 40), (36, 10), (39, 90), (61, 5), (60, 45)]
group_sizes = [10, 10, 10, 50, 20]
total_pedestrians = sum(group_sizes)
initial_positions = np.zeros((total_pedestrians, 2))

# 分配各组的起始点和途经点
waypoints_groups = [
    np.array([[40, 45], [75, 55], [85, 95]]),
    np.array([[36, 40], [75, 55], [85, 95]]),
    np.array([[47, 45], [75, 55], [85, 95]]),
    np.array([[61, 45], [75, 55], [85, 95]]),
    np.array([[75, 55], [85, 95]]),
]

# 初始化位置
index = 0
group_indices = []
for group_id, (start_point, size) in enumerate(zip(start_points, group_sizes)):
    initial_positions[index:index + size] = np.tile(start_point, (size, 1))
    group_indices.extend([group_id] * size)

    index += size
# 添加随机位置的行人

# 模拟参数
num_steps = 50000
positions_time = np.zeros((num_steps, total_pedestrians * 2))
current_indices = np.zeros(total_pedestrians, dtype=int)

# 加载背景图
img = mpimg.imread('MI.png')
'''
可以修改的部分↑
'''


plt.figure(figsize=(10, 8))

tag = 0
completed_count = 0
# 模拟过程
for step in range(num_steps):
    if completed_count == 100:
        break
    initial_positions, current_indices = update_positions(initial_positions, waypoints_groups, current_indices)
    completed_count = count_completed_pedestrians(current_indices, waypoints_groups)
    positions_time[step, :] = initial_positions.flatten()
    if step % 100 == 0:
        plt.clf()
        plt.imshow(img, extent=[0, 100, 0, 100])
        for waypoints in waypoints_groups:
            plt.scatter(waypoints[:, 0], waypoints[:, 1], c='red', marker='o')
        plt.scatter(initial_positions[:, 0], initial_positions[:, 1], c='blue', s=3)
        plt.pause(0.00001)
        print(step)
plt.show()

print(completed_count)
column_names = [f"x{i+1}{j+1},y{i+1}{j+1}" for i in range(total_pedestrians) for j in range(2)]
df = pd.DataFrame(positions_time[:step], columns=column_names)
df.to_csv("pedestrians_positions_MI.csv", header=False, index=False, sep=' ')