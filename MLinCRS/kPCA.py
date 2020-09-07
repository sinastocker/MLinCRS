import numpy as np


# kPCA
def kPCA(K):
    """
    Function to perform a kernel PCA (kPCA).

    Input:
    ------
    K : np.array
        Kernel matrix for which you want to
        perform a kPCA

    Returns:
    --------
    x1 : np.array
        PC1
    x2 : np.array
        PC2
    x3 : np.array
        PC3
    """
    # centralize K - this is the equivalent of the mean shift
    one = 1.0/K.shape[0] * np.ones((K.shape[0], K.shape[1]))
    KK = K - np.matmul(one, K) - np.matmul(K, one) + np.linalg.multi_dot([one, K, one])

    # compute eigenvectors
    v, F = np.linalg.eigh(KK)

    # Sort eigenvalues to get the largest ones
    idx = np.argsort(v)[::-1]
    # coordinates are projections along first two eigenvectors
    x1 = np.matmul(-K, F[:, idx[0]])
    x2 = np.matmul(-K, F[:, idx[1]])
    x3 = np.matmul(-K, F[:, idx[2]])

    return x1, x2, x3
