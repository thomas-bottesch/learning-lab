from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
        use_batchnorm=False,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.use_batchnorm = use_batchnorm
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################

        self.params = {}
        C, H, W = input_dim
        std = weight_scale

        H_out = H // 2
        W_out = W // 2

        i = 0
        self.params["W1"] = std * np.random.randn(
            num_filters, C, filter_size, filter_size
        )
        self.params["b1"] = np.zeros(num_filters)
        if self.use_batchnorm:
            self.params[f"bn_gamma{i+1}"] = np.ones(num_filters)
            self.params[f"bn_beta{i+1}"] = np.zeros(num_filters)
        i += 1

        self.params["W2"] = std * np.random.randn(
            np.prod([num_filters, H_out, W_out]), hidden_dim
        )
        self.params["b2"] = np.zeros(hidden_dim)
        if self.use_batchnorm:
            self.params[f"bn_gamma{i+1}"] = np.ones(hidden_dim)
            self.params[f"bn_beta{i+1}"] = np.zeros(hidden_dim)

        self.params["W3"] = std * np.random.randn(hidden_dim, num_classes)
        self.params["b3"] = np.zeros(num_classes)

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        if not self.use_batchnorm:
            x_out, c1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
            x_out, c2 = affine_relu_forward(x_out, W2, b2)
        else:
            i = 0
            x_out, c1 = conv_bn_relu_pool_forward(
                X,
                W1,
                b1,
                self.params[f"bn_gamma{i+1}"],
                self.params[f"bn_beta{i+1}"],
                conv_param,
                pool_param,
                self.bn_params[i],
            )
            i += 1
            x_out, c2 = affine_bn_relu_forward(
                x_out,
                W2,
                b2,
                self.params[f"bn_gamma{i+1}"],
                self.params[f"bn_beta{i+1}"],
                self.bn_params[i],
            )
        x_out, c3 = affine_forward(x_out, W3, b3)

        scores = x_out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        loss, d_X3 = softmax_loss(x_out, y)

        loss += 0.5 * self.reg * np.sum(self.params["W3"] * self.params["W3"])
        loss += 0.5 * self.reg * np.sum(self.params["W2"] * self.params["W2"])
        loss += 0.5 * self.reg * np.sum(self.params["W1"] * self.params["W1"])

        dX_2, dW_3, db_3 = affine_backward(d_X3, c3)
        if not self.use_batchnorm:
            dX_1, dW_2, db_2 = affine_relu_backward(dX_2, c2)
            dX, dW_1, db_1 = conv_relu_pool_backward(dX_1, c1)
        else:
            for i in range(2):
                grads[f"bn_gamma{i+1}"] = np.zeros_like(self.params[f"bn_gamma{i+1}"])
                grads[f"bn_beta{i+1}"] = np.zeros_like(self.params[f"bn_beta{i+1}"])

            i = 1
            dX_1, dW_2, db_2, dgamma_2, dbeta_2 = affine_bn_relu_backward(dX_2, c2)
            grads[f"bn_gamma{i+1}"] += dgamma_2
            grads[f"bn_beta{i+1}"] += dbeta_2

            i -= 1
            dX, dW_1, db_1, dgamma_1, dbeta_1 = conv_bn_relu_pool_backward(dX_1, c1)
            grads[f"bn_gamma{i+1}"] += dgamma_1
            grads[f"bn_beta{i+1}"] += dbeta_1

        grads["W3"] = dW_3
        grads["W2"] = dW_2
        grads["W1"] = dW_1
        grads["b1"] = db_1
        grads["b2"] = db_2
        grads["b3"] = db_3

        grads["W3"] += self.reg * self.params["W3"]
        grads["W2"] += self.reg * self.params["W2"]
        grads["W1"] += self.reg * self.params["W1"]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
