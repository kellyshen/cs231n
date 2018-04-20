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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(len(X)): # N
    #num = 0
    #denom = 0
    s = np.dot(X[i], W) # C
    s_max = np.max(s)
    exps = np.exp(s - s_max)
    softs = exps / np.sum(exps)
    #print softs
    for j in range(len(softs)):
      if y[i] == j:
        loss += -np.log(softs[j])
        dW[:, j] -= X[i]
        #num = np.exp(s[j])
      #denom += np.exp(s[j])
    #soft = np.log(num / denom)
    #loss += -soft
    dW += np.dot(X[i].reshape(len(X[i]), 1), softs.reshape(1, len(softs)))
  loss /= len(X)
  loss += reg * np.sum(W * W)#reg * np.sum(np.multiply(W, W))
  dW /= len(X)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W) # N * C
  scores_max = np.max(scores, axis=1) # N, 
  nums = np.exp(scores - scores_max.reshape(len(scores_max), 1)) # N * C 
  denoms = np.sum(nums, axis=1)
  softs = nums / denoms.reshape(len(denoms), 1) # N * C
  #print np.sum(softs, 1)
  trues = softs[[i for i in range(len(X))], y]
  #print np.sum(np.log(trues))
  #print -1.0/len(X) * np.sum(np.log(trues))
  loss = -1.0/len(X) * np.sum(np.log(trues)) + reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  N, D = X.shape
  multipliers = np.zeros(scores.shape) # N * C
  multipliers[[i for i in range(N)], y] = -1
  dW = (X.T.dot(multipliers) + X.T.dot(softs))/ N + 2 * reg * W 

  return loss, dW

