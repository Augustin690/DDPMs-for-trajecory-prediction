import numpy as np
import matplotlib.pyplot as plt
import utils

"""Task2.3: In this script, we demonstrate the similarity of Diffusion Maps and Fourier analysis using a periodic dataset.
We need functions defined in utils.py for this script.
"""

# TODO: Create a periodic dataset with the details described in the task-sheet
data = np.loadtxt('pedestrians_positions_MI.csv', delimiter=' ')
# TODO: Visualize data-set
plt.scatter(data[:, 0], data[:, 1], s=1)
plt.title('dataset')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# TODO: Compute eigenfunctions associated to the largest eigenvalues using function diffusion_map() implemented in utils.py
lambdas, phi = utils.diffusion_map(data)
# TODO: Plot plot the first non-constant eigenfunction φ1 against the other eigenfunctions
phi_1 = phi[:, 1]
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i, ax in enumerate(axes.flat, start=2):
    ax.scatter(phi_1, phi[:, i], s=2)
    ax.set_xlabel('φ1')
    ax.set_ylabel('φ{}'.format(i))
    ax.set_title(f'φ1 vs φ{i}')
plt.show()

