from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        
        A1, cache_1 = affine_relu_forward(X, W1, b1)
        Z2, cache_2 = affine_forward(A1, W2, b2)
        scores = Z2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dZ2 = softmax_loss(scores, y)
        loss+=0.5*self.reg*(np.sum(W1*W1) + np.sum(W2*W2))
        X2 = np.copy(X).reshape(X.shape[0], -1)
        grads['W2'] = np.dot(A1.T, dZ2) + self.reg * W2
        grads['b2'] = np.sum(dZ2, axis=0)
        grads['W1'] = np.dot(X2.T, np.dot(dZ2, W2.T)) + self.reg * W1
        grads['b1'] = np.sum(np.dot(dZ2, W2.T), axis=0)

        
        grads_2 = affine_backward(dZ2, cache_2)#(A1, W2, b2))
        grads['W2'] = grads_2[1] + self.reg * W2
        grads['b2'] = grads_2[2]
        # Z1, _ = affine_forward(X, W1, b1)
        grads_1 = affine_relu_backward(grads_2[0], cache_1)#((X, W1, b1), Z1))
        grads['W1'] = grads_1[1] + self.reg * W1
        grads['b1'] = grads_1[2] 

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        #print("1")
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.hidden_dims = hidden_dims
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        #print("hi")
        #asdfasd
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        for i in range(1, self.num_layers+1):
            if i == 1:
                curr_dim = self.hidden_dims[0]
                self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, curr_dim))
                self.params['b1'] = np.zeros(curr_dim)
                if self.normalization == 'batchnorm':
                    self.params['alpha1'] = np.ones(curr_dim)
                    self.params['beta1'] = np.zeros(curr_dim)
            else:
                prev_dim = self.hidden_dims[i-2]  
                if i == self.num_layers:
                    curr_dim = num_classes
                else:
                    curr_dim = self.hidden_dims[i-1]
                self.params['W' + str(i)] = np.random.normal(0, weight_scale, (prev_dim, curr_dim))
                self.params['b' + str(i)] = np.zeros(curr_dim)
                if self.normalization == 'batchnorm':
                    self.params['alpha' + str(i)] = np.ones(curr_dim)
                    self.params['beta' + str(i)] = np.zeros(curr_dim)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

        #print(self.params.keys())

    def loss(self, X, y=None):
        #print("start loss")
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            #print('hello')
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        caches = []
        a = np.copy(X)
        for i in range(1, self.num_layers+1): # all but last layer
            #print('iter ', i)
            W = self.params['W' + str(i)]
            b = self.params['b' + str(i)]
            #z, cache_aff = affine_forward(z, w, b)
            #if self.normalization=='batchnorm':

            #a, cache_relu = relu_forward(z)
            if i == self.num_layers:
                z, cache = affine_forward(a, W, b) 
            else: 
                a, cache = affine_relu_forward(a, W, b)
                if self.use_dropout:
                    #print('hello')
                    a, cache2 = dropout_forward(a, self.dropout_param)
                    # print('i is ', i)
                    # print('cache2')
                    # print(cache2)
                    # print("****")
                    cache = [cache, cache2]
                    # print(cache)
                    # print("******")

            caches.append(cache)
        # print("******")
        # print(len(caches))
        # print('caches[0]')
        # print(cachess[0])
        # print('caches[1]')
        # print(caches[1])
        scores = z
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dz = softmax_loss(scores, y)
        reg_total = 0

        for i in range(self.num_layers, 0, -1): # this includes the final layer
            if i == self.num_layers:
                #print("i is ", i)
                #print(len(caches[i-1]))
                #print("****")
                grads_curr = affine_backward(dz, caches[i-1])#(A1, W2, b2))
            else:
                if self.use_dropout:
                    #print('in dropout')
                    #print("i is ", i)
                    #print(len(caches[i-1]))
                    #print(len(caches[i-1][0]))
                    #print("****")
                    temp = dropout_backward(grads_curr[0], caches[i-1][1])
                    grads_curr = affine_relu_backward(temp, caches[i-1][0])
                else:
                    grads_curr = affine_relu_backward(grads_curr[0], caches[i-1])
            W = self.params['W' + str(i)]
            reg_total+=np.sum(W*W)
            grads['W' + str(i)] = grads_curr[1] + self.reg * W
            grads['b' + str(i)] = grads_curr[2]

        loss+=0.5*self.reg*reg_total

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
