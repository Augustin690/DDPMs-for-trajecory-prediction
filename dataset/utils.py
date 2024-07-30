import scipy as sp
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt

""" This script contains the implementation of the diffusion map algorithm. 
"""

def create_distance_matrix(X, max_distance=200):
    """ Compute a sparse distance matrix using scipy.spatial.KDTree. Set max_distance as 200.

    Args:
        X (npt.NDArray[np.float]): Data matrix.
        max_distance (int, optional): Computes a distance matrix leaving as zero any distance greater than max_distance. Defaults to 200.

    Returns:
        npt.NDArray[np.float]: Distance Matrix  
        output shape = (np.shape(D)[0], np.shape(D)[0]). 
    """
    # TODO: Implement method
    # Hints: using scipy.spatial.KDTree, set max_distance as 200, you may have to use .toarray() to the array you are returning!)
    tree = sp.spatial.KDTree(X)
    distance_matrix = tree.sparse_distance_matrix(tree, max_distance)
    return distance_matrix.toarray()

def set_epsilon(p, distance_matrix):
    """ Set scalar epsilon as 'p' % of the diameter of the dataset.
    (Step 2 of the algorithm mentioned in the worksheet.)

    Args:
        p (np.float64): percentage.
        distance_matrix (npt.NDArray[np.float]): Distance matrix.

    Returns:
        np.float64: returns epsilon.
    """
    # TODO: Implement method (Hint: p is a float between 1-100, you have to divide by 100)
    return (p/100) * np.max(distance_matrix)


def create_kernel_matrix(D, eps):
    """Create the Kernel matrix.

    Args:
        D (npt.NDArray[np.float]): Distance matrix
        eps (np.float64): epsilon.

    Returns:
        npt.NDArray[np.float]: Kernel matrix. (output shape = (np.shape(D)[0], np.shape(D)[0]))
    """
    # TODO: Form the kernel matrix W (Step 3 of the algorithm from the worksheet)
    W = np.exp(-D**2 / eps)
    # TODO: Form the diagonal normalization matrix (Step 4 of the algorithm from the worksheet)
    P_ii = np.sum(W, axis=1) # sum j, axis=1
    # TODO: Normalize W to form the kernel matrix K (Step 5 of the algorithm from the worksheet)
    P_inverse = np.diag(1/P_ii)
    K = P_inverse @ W @ P_inverse
    return K


def diffusion_map(X, n_eig_vals=5):
    """ Implementation of the diffusion map algorithm.
        Please refer to the algorithm in the worksheet for the following.
        The step numbers in the following refer to the steps of the algorithm in the worksheet.

    Args:
        X (npt.NDArray[np.float]): Data matrix (each row represents one data point)
        n_eig_vals (int, optional): The number of eigenvalues and eigenvectors of the Laplace-Beltrami operator defined on the manifold close to the data to be computed. Default is 10.

    Returns:
        tuple(npt.NDArray[np.float], npt.NDArray[np.float]): eigenvalues, eigenvector of the Laplace-Beltrami operator
        output shapes: (n_eig_vals + 1, ), (np.shape(X)[0], n_eig_vals + 1)
    """

    # TODO: Compute distance matrix. Use method create_distance_matrix(..) defined in this script. (Step 1 from the algorithm in the worksheet)
    distance_matrix = create_distance_matrix(X)
    # TODO: Use function set_epsion(.., ..) defined in this script to set epsilon to 5% of the diameter of the dataset (Step 1 from the algorithm in the worksheet).
    epsilon = set_epsilon(5, distance_matrix)
    # TODO: Form Kernel matrix K. Use function create_kernel_matrix(.., ..) defined in this script. (Steps 3-5 from the algorithm in the worksheet) 
    K = create_kernel_matrix(distance_matrix, epsilon)
    # TODO: Form the diagonal normalization matrix (Step 6 from the algorithm in the worksheet)
    Q_ii =  np.sum(K, axis=1)
    qsqr = np.sqrt(Q_ii)
    mqsqr  = 1 / qsqr
    Q_inverse_sqrt = np.diag(mqsqr)
    # TODO: Form symmetric matrix T_hat (Step 7 from the algorithm in the worksheet)
    T_hat = Q_inverse_sqrt @ K @ Q_inverse_sqrt
    # TODO: Find the L + 1 largest eigenvalues and the corresponding eigenvectors of T_hat (Step 8 from the algorithm in the worksheet)
    eigvals, eigvecs = eigh(T_hat)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    # TODO: Compute the eigenvalues of T_hat^(1/ε) in DESCENDING ORDER (Hint: You can use np.flip(..))!! (Step 9 from the algorithm in the worksheet)
    lambdas = eigvals[:n_eig_vals+1] ** (1/epsilon)
    # TODO: Compute the eigenvectors of the matrix T (Hint: You can use np.flip(..) with an appropriate axis) (Step 10 from the algorithm in the worksheet)
    phi = Q_inverse_sqrt @ eigvecs[:, :n_eig_vals+1]
    return np.sqrt(lambdas), phi

