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

  num_samples = X.shape[0]
  num_classes = W.shape[1]
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  ############################################################################# 

  for i in xrange(num_samples):
      
      f_i = X[i].dot(W)
      f_i -= np.max(f_i) 
      
      denom = np.sum(np.exp(f_i))
      probs = np.exp(f_i)/denom 
      
      loss += -np.log(probs[y[i]])
      
      for k in xrange(num_classes):
           p_k = probs[k]     
           dW[:,k] += (p_k - (y[i] == k)) * X[i]
#==============================================================================
#       for j in xrange(num_classes):
#           if y[i] == j: #if correct label
#               num = np.exp(score[i,j]-C)
#                             
#           probForEachClass = np.exp(score[i,j]-C)
#           dW[:,j] = probForEachClass
#           denom += probForEachClass
#             
#       dW[i] /= denom
#       dW[i,y[i]] -= 1
#   
#       loss += -np.log(num/denom)
#==============================================================================
      
        
  loss /= num_samples
  dW /= num_samples
  
  #regularization 
  loss += 0.5* reg * np.sum(W * W)
  dW += reg * W
  

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
  num_samples = X.shape[0]
  
  scores = X.dot(W) # N by C
  scores -= np.max(scores, axis=1).reshape(num_samples, -1)
  
  probs = np.exp(scores)
  
  sum_probs = np.sum(probs, axis=1) #N dim
  norm_prob = probs/sum_probs.reshape(num_samples,-1) #reshape for broadcasting
  
  corr_class = norm_prob[np.arange(num_samples), y] #N dim
  
  loss = np.sum(-np.log(corr_class))
  
  temp = norm_prob
  temp[np.arange(num_samples), y] -= 1 #subtract one from each correct class prob
       
  dW = X.T.dot(temp)
  
  loss /= num_samples
  dW /= num_samples
  
  #regularization 
  loss += 0.5* reg * np.sum(W * W)
  dW += reg * W  

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

