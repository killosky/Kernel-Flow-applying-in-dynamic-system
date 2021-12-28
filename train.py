
import numpy as np
import torch
import random
import scipy
import matplotlib.pyplot as plt
from data import get_dataset_slide
from args import get_args



def sample(N_f, N_c, i_Nf, args):
    '''
    :param N_f:
    :param N_c:
    :return: the sampling matrix of dimension (Nc,Nf)
    '''
    sampling = random.sample(range(N_f), N_c)
    i_Nc = i_Nf[sampling]

    Pi = np.zeros((N_c, N_f), dtype=float)

    for i in range(N_c):
        Pi[i, sampling[i]] = 1

    return i_Nc, Pi

def generate_Nf(X, args):
    '''
    :param X:
    :return: the index i_Nf of the training process such that the distance of the point less than 1e-4
    '''
    N = X.shape[0]
    i_Nf = np.array(random.sample(range(N), args.sample), dtype=int)
    for i in range(args.sample):
        for j in np.arange(i+1, args.sample):
            if scipy.linalg.norm(X[i_Nf[i]] - X[i_Nf[j]]) < 1e-4:
                i_Nf = np.delete(i_Nf, np.where(i_Nf == i))
                break

    return i_Nf

def K1(x1, x2, args):
    '''
    :param x1:
    :param x2:
    :param args:
    :return: the kernel function of the system
    '''
    # return np.exp(-args.gamma*np.sum(np.square(x1-x2)))
    return np.exp(-args.gamma*(scipy.linalg.norm(x1-x2)**2))

def get_K(x1, x2, args):
    n = x1.shape[0]
    m = x2.shape[0]
    Theta = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            Theta[i, j] = K1(x1[i, :], x2[j, :], args)
    return Theta

def gradK(x, args):
    n = x.shape[0]
    m = x.shape[1]
    Delta = np.zeros((m, n, n))

    for i in range(n):
        for j in range(n):
            Delta[:, i, j] = np.exp(-args.gamma * (scipy.linalg.norm(x[i, :]-x[j, :])**2)) * (-2) * args.gamma * (x[i, :] - x[j, :])
    return Delta


def train(args):
    '''
    :param args: the parameter of the process
    :return: return the train result
    '''

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    loss = []
    test_loss = []

    # initial the train data
    data = get_dataset_slide(seed=args.seed, input_slide=args.input_slide, output_slide=args.output_slide)
    X = np.array(data['input'])
    Y = np.array(data['output'])
    X_test = np.array(data['test_input'])
    Y_test = np.array(data['test_output'])


    N = X.shape[0]
    N_test = X_test.shape[0]
    dimx = X.shape[1]

    # train the model
    for n in range(args.total_step):

        i_Nf = generate_Nf(X, args)
        N_f = i_Nf.shape[0]
        N_c = N_f//2
        i_Nc, Pi = sample(N_f, N_c, i_Nf, args)

        # structure the kernel flow matrix Theta
        Theta = get_K(X[i_Nf], X[i_Nf], args)
        Delta = gradK(X[i_Nf], args)

        invTheta = scipy.linalg.inv(Theta)
        invPiThetaPiT = scipy.linalg.inv(Pi @ Theta @ Pi.T)

        Y_hat = invTheta @ Y[i_Nf]
        z_hat = Pi.T @ invPiThetaPiT @ Pi @ Y[i_Nf]
        rho = 1 - (Y[i_Nc].T @ invPiThetaPiT @ Y[i_Nc]).trace() / (Y[i_Nf].T @ Y_hat).trace()

        g = np.zeros((dimx, N_f))

        for s in range(dimx):
            g[s, :] = (2 * ((1 - rho) * (np.multiply(Y_hat, (Delta[s, :, :].squeeze() @ Y_hat))).sum(1) - (np.multiply(z_hat, (Delta[s, :, :].squeeze() @ z_hat))).sum(1))) / ((Y[i_Nf].T @ Y_hat).trace())
        G = g @ invTheta @ get_K(X[i_Nf], X, args)
        G_test = g @ invTheta @ get_K(X[i_Nf], X_test, args)

        p = args.learning_rate
        epsilon = np.min(np.divide(p * scipy.linalg.norm(X, axis=1), scipy.linalg.norm(G, axis=0)))
        X += epsilon * G.T
        X_test += epsilon * G_test.T

        if not n % args.print_every:
            Y_result = get_K(X, X[i_Nf], args) @ scipy.linalg.inv(get_K(X[i_Nf], X[i_Nf], args)) @ Y[i_Nf]
            error = scipy.linalg.norm(Y-Y_result)**2 / N
            Y_test_result = get_K(X_test, X[i_Nf], args) @ scipy.linalg.inv(get_K(X[i_Nf], X[i_Nf], args)) @ Y[i_Nf]
            test_error = scipy.linalg.norm(Y_test-Y_test_result)**2 / N_test

            loss.append([error, n])
            test_loss.append([test_error, n])

            print("\r n={}, error={}, test_error={}".format(n, error, test_error), end="")

    loss = np.array(loss)
    test_loss = np.array(test_loss)

    plt.figure(1)
    plt.plot(loss[:, 1], loss[:, 0])
    plt.savefig('loss.png')

    plt.figure(2)
    plt.plot(test_loss[:, 1], test_loss[:, 0])
    plt.savefig('test_loss.png')

    return loss, test_loss


if __name__ == "__main__":
    args = get_args()
    print(train(args))

