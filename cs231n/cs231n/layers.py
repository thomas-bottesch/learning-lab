from builtins import range
import numpy as np

# import numexpr as ne # ~~DELETE LINE~~


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    _x = x.reshape((x.shape[0], np.prod(x.shape[1:])))
    out = np.dot(_x, w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################

    _x = x.reshape((x.shape[0], np.prod(x.shape[1:])))

    # dz / dx = w
    # dL / dz (upstream gradient)
    # dL / dx = (dL / dz).dot(w.T)
    dx = dout.dot(w.T).reshape(x.shape)

    # dz / dw = x
    # dL / dz (upstream gradient)
    # dL / dw = x.T.dot((dL / dz))
    dw = _x.T.dot(dout)

    # dL / db = sum{1...N} dL/z_i
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################

    out = x.copy()
    out[out < 0] = 0

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################

    dx = dout.copy()
    dx[x <= 0] = 0

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # Compute output
        mu = x.mean(axis=0)

        # Center x
        xc = x - mu

        # Calculate variance
        var = np.mean(xc**2, axis=0)

        # Calculate std
        std = np.sqrt(var + eps)

        # Normalize x
        xn = xc / std

        # Calculate the batch norm
        out = gamma * xn + beta

        cache = (x, gamma, xc, std, xn)

        # Update running average of mean
        running_mean *= momentum
        running_mean += (1 - momentum) * mu

        # Update running average of variance
        running_var *= momentum
        running_var += (1 - momentum) * var

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        std = np.sqrt(running_var + eps)
        xn = (x - running_mean) / std
        out = gamma * xn + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################

    x, gamma, xc, std, xn = cache

    N = x.shape[0]
    dbeta = dout.sum(axis=0)
    dgamma = np.sum(xn * dout, axis=0)
    dxn = gamma * dout
    dxc = dxn / std
    dstd = -np.sum((dxn * xc) / (std * std), axis=0)
    dvar = 0.5 * dstd / std
    dxc += (2.0 / N) * xc * dvar
    dmu = np.sum(dxc, axis=0)
    dx = dxc - dmu / N

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################

    x, gamma, xc, std, xn = cache
    N = x.shape[0]

    dbeta = dout.sum(axis=0)
    dgamma = np.sum(xn * dout, axis=0)

    dx = (
        (1.0 / N)
        * gamma
        * (std**-1)
        * (N * dout - dout.sum(axis=0) - xn * (dout * xn).sum(axis=0))
    )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################

    # Compute mean for each data point (across features)
    mu = np.mean(x, axis=1, keepdims=True)  # (N, 1)
    # Center the input by subtracting mean
    xc = x - mu  # (N, D)
    # Compute variance for each data point (across features)
    var = np.mean(xc**2, axis=1, keepdims=True)  # (N, 1)
    # Compute standard deviation for each data point
    std = np.sqrt(var + eps)  # (N, 1)
    # Normalize the centered input
    xn = xc / std  # (N, D)
    # Scale and shift the normalized input
    out = gamma * xn + beta  # broadcast (N, D)
    # Store values needed for backward pass
    cache = (x, gamma, xc, std, xn)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################

    x, gamma, xc, std, xn = cache
    N, D = x.shape
    # Gradients for gamma and beta (broadcasted)
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(xn * dout, axis=0)
    # Gradient w.r.t. normalized input
    dxn = dout * gamma
    # Layer norm: statistics are per data point (across features)
    # So, axis=1 for mean/var/std
    dxc = dxn / std
    dstd = -np.sum((dxn * xc) / (std**2), axis=1, keepdims=True)
    dvar = 0.5 * dstd / std
    dxc += (2.0 / D) * xc * dvar
    dmu = np.sum(dxc, axis=1, keepdims=True)
    dx = dxc - dmu / D

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################

        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    pad = conv_param["pad"]
    stride = int(conv_param["stride"])
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, H_out, W_out))

    x_p = np.pad(
        x,
        (
            (0, 0),  # do not pad N
            (0, 0),  # do not pad C
            (pad, pad),  # pad H
            (pad, pad),  # pad W
        ),
        mode="constant",
    )

    for i in range(N):
        for j in range(F):
            for k in range(H_out):
                h_pos = k * stride
                for l in range(W_out):
                    w_pos = l * stride

                    # w[j][:, :, :]                                     -   matrix of shape C, HH, WW
                    # x_p[i][:, h_pos : h_pos + HH, w_pos : w_pos + WW] -   matrix of shape C, HH, WW
                    # the multiply multiplys every element of the two matrices
                    # the sum sums all elements from all dimensions up
                    # This is basically w*x + b where w is one specific filter
                    # and x is actually only a small region within X with the
                    # same size as the filter
                    out[i][j][k][l] += (
                        np.sum(
                            x_p[i][:, h_pos : h_pos + HH, w_pos : w_pos + WW]
                            * w[j][:, :, :]
                        )
                        + b[j]
                    )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################

    x, w, b, conv_param = cache
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    pad = conv_param["pad"]
    stride = int(conv_param["stride"])
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)

    x_p = np.pad(
        x,
        (
            (0, 0),  # N
            (0, 0),  # C
            (pad, pad),  # H
            (pad, pad),  # W
        ),
        mode="constant",
    )
    dx_p = np.pad(
        dx,
        (
            (0, 0),  # N
            (0, 0),  # C
            (pad, pad),  # H
            (pad, pad),  # W
        ),
        mode="constant",
    )

    for i in range(N):
        for j in range(F):
            for k in range(H_out):
                h_pos = k * stride
                for l in range(W_out):
                    w_pos = l * stride

                    dx_p[i, :, h_pos : h_pos + HH, w_pos : w_pos + WW] += (
                        w[j, :, :, :] * dout[i, j, k, l]
                    )
                    dw[j, :, :, :] += (
                        x_p[i, :, h_pos : h_pos + HH, w_pos : w_pos + WW]
                        * dout[i, j, k, l]
                    )
                    db[j] += dout[i, j, k, l]

        dx[i, :, :, :] = dx_p[i, :, pad : pad + H, pad : pad + W]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################

    N, C, H, W = x.shape

    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)

    out = np.zeros((N, C, H_out, W_out))
    for i in range(N):
        for k in range(H_out):
            h_pos = k * stride
            for l in range(W_out):
                w_pos = l * stride
                out[i, :, k, l] = np.max(
                    x[i, :, h_pos : h_pos + pool_height, w_pos : w_pos + pool_width],
                    axis=(1, 2),
                )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################

    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)

    dx = np.zeros_like(x)

    for i in range(N):
        for k in range(H_out):
            h_pos = k * stride
            for l in range(W_out):
                w_pos = l * stride
                for c in range(C):
                    window = x[
                        i, c, h_pos : h_pos + pool_height, w_pos : w_pos + pool_width
                    ]
                    max_idx = np.unravel_index(np.argmax(window), window.shape)
                    dx[i, c, h_pos : h_pos + pool_height, w_pos : w_pos + pool_width][
                        max_idx
                    ] = dout[i, c, k, l]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = x.shape
    # The transpose(0, 2, 3, 1) operation changes the axes order from
    # (N, C, H, W) to (N, H, W, C):
    # 0: N (batch)
    # 2: H (height)
    # 3: W (width)
    # 1: C (channels)

    # After transposing, .reshape(-1, C) flattens the first three dimensions
    # (N, H, W) into one, resulting in a 2D array of shape (NHW, C).
    x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)

    # Call the standard batchnorm_forward using the flattened x.
    out_flat, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)

    # Reshape and transpose dx back to the original (N, C, H, W) shape.
    out = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = dout.shape
    # The transpose(0, 2, 3, 1) operation changes the axes order from
    # (N, C, H, W) to (N, H, W, C):
    # 0: N (batch)
    # 2: H (height)
    # 3: W (width)
    # 1: C (channels)

    # After transposing, .reshape(-1, C) flattens the first three dimensions
    # (N, H, W) into one, resulting in a 2D array of shape (NHW, C).
    dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    # Call the standard batchnorm_backward using the flattened dout and cache.
    dx_flat, dgamma, dbeta = batchnorm_backward(dout_flat, cache)
    # Reshape and transpose dx back to the original (N, C, H, W) shape.
    dx = dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    N, C, H, W = x.shape
    x_group = x.reshape(N, G, C // G, H, W)
    # Normalize over (C//G, H, W) for each group
    mean = np.mean(x_group, axis=(2, 3, 4), keepdims=True)
    var = np.var(x_group, axis=(2, 3, 4), keepdims=True)
    std = np.sqrt(var + eps)
    x_groupnorm = (x_group - mean) / std
    x_norm = x_groupnorm.reshape(N, C, H, W)
    out = gamma * x_norm + beta
    cache = (G, x, x_norm, mean, std, gamma, beta)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    G, x, x_norm, mean, std, gamma, beta = cache
    N, C, H, W = x.shape
    # Gradients for gamma and beta
    dgamma = np.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

    # Gradient w.r.t. normalized input
    dx_norm = dout * gamma
    # Reshape for group computation
    dx_group = dx_norm.reshape(N, G, C // G, H, W)
    x_group = x.reshape(N, G, C // G, H, W)
    m = C // G * H * W

    # Backprop through normalization
    dvar = np.sum(
        dx_group * (x_group - mean) * -0.5 * (std**-3), axis=(2, 3, 4), keepdims=True
    )
    dmean = (
        np.sum(dx_group * -1 / std, axis=(2, 3, 4), keepdims=True)
        + dvar * np.sum(-2 * (x_group - mean), axis=(2, 3, 4), keepdims=True) / m
    )
    dx_group_out = dx_group / std + dvar * 2 * (x_group - mean) / m + dmean / m

    dx = dx_group_out.reshape(N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################

    num_train = x.shape[0]
    scores = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(scores), axis=1, keepdims=True)
    log_probs = scores - np.log(Z)
    probs = np.exp(log_probs)
    loss = -np.sum(log_probs[np.arange(num_train), y]) / num_train

    dx = probs.copy()
    dx[np.arange(num_train), y] -= 1
    dx = dx / num_train
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
