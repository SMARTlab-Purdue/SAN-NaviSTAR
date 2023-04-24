import torch
import torch.nn as nn
from torch.utils import data
from matplotlib import pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigs, eigsh

#colors = ['green', 'blue', 'red', 'purple', 'black', 'yellow', 'orange', 'pink', 'indigo', 'gray',
          #'lightblue', 'lightgreen', 'lightyellow', 'tomato', 'gold', 'olive', 'white', 'violet', 'deepskyblue', 'khaki']


class My_loss(nn.Module):   
    def __init__(self):
        super().__init__()

    def forward(self, x, y, agent):
        loss = 0
        criterion = nn.MSELoss()

        for i in range(len(agent)):
            true_x = x[i, :agent[i], :, :]
            true_y = y[i, :agent[i], :, :]

            # sum_y = torch.sum(true_y.detach(), dim = -1, keepdim = True)
            # index = sum_y.repeat(1, 1, 2)
            # true_x = true_x * index

            loss = loss + criterion(true_x, true_y)

        return loss


class My_dataset(data.Dataset):
    def __init__(self, x, y, agent, adj):
        super().__init__()

        self.x = x
        self.y = y
        self.agent = agent
        self.adj = adj

    def __getitem__(self, index):
        result = self.x[index]
        label = self.y[index]
        agent = self.agent[index]
        adj = self.adj[index]

        return result, label, agent, adj

    def __len__(self):
        return len(self.x)




def remove_zero(x, y): 
    x = x.numpy()
    y = y.numpy()
    y_sum = np.sum(y, axis = 1)
    indice = np.where((y_sum != 0))[0]
    x = x[indice]
    y = y[indice]

    return x, y

def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    # print(W)
    # print('#############')

    D = np.diag(np.sum(W, axis=1))
    D_ = D ** 0.5
    D_inv = np.linalg.inv(D)
    D_inv_ = D_inv ** 0.5
    L = np.identity(W.shape[0]) - np.matmul(np.matmul(D_inv_, W), D_)

    #L = D - W

    try:
        # lambda_max = eigs(L, k = 1, which='LR')[0].real
        lambda_max, _ = eigsh(L, 1, which='LM')

    except Exception as e:
        # print(e)
        # print(L)
        lambda_max = eigsh(L, k = 1, which='LM')[0].real


    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials
