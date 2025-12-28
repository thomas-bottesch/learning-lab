from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)
        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)
        loss -= logp[y[i]]  # negative log probability is the loss

        # Gradient calculation
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += (p[j] - 1) * X[i]
            else:
                dW[:, j] += p[j] * X[i]

    # Average loss and gradient, add regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################

    XW = X.dot(W).T
    scores = XW - np.max(XW, axis=1)
    p = np.exp(scores)
    p /= np.sum(p, axis=0)
    logp = np.log(p)
    loss = -np.sum(logp.T[np.arange(len(y)), y])
    loss = loss / num_train + reg * np.sum(W * W)

    # for every element in y determine the position of the correct class in
    # num_classes. This will also be the correct position where to add into the
    # vector dW
    D = np.searchsorted(range(num_classes), y)

    # A we know ftom the previous implementation this dW[:, j] += p[j] * X[i]
    # happens for every sample no matter what. Only if the label of the sample
    # equals the desired label there is something additional to do
    # so we do:
    dW = p.dot(X)

    # Now the additional part. We have to subtract X[i] from W[i] in line D
    # We are basically doing this dW[:, j] += -X[i]
    np.add.at(dW, D, -X)

    dW = dW.T / num_train + 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    return loss, dW
