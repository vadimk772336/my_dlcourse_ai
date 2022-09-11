import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    predictions -= np.max(predictions, axis=-1, keepdims=True)
    pred_exp = np.exp(predictions)
    probs = pred_exp/np.sum(pred_exp, axis=-1, keepdims=True)
    
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    if (probs.ndim == 1):
        loss = -np.log(probs[target_index])
    else:
        rows = np.arange(len(target_index)) 
        cols = target_index.reshape(1, -1)
        loss = -np.mean(np.log(probs[rows, cols]))
        
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    
    probs = softmax(predictions.copy())
    loss = cross_entropy_loss(probs, target_index)
    
    E = np.zeros(predictions.shape)
    batch_size = 1
    if (predictions.ndim == 1):
        E[target_index] = 1
    else:
        batch_size = len(target_index)
        rows = np.arange(batch_size) 
        cols = target_index.reshape(1, -1)
        E[rows, cols] = 1

    dprediction = (probs - E)/batch_size
    
    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength*np.sum(W*W)
    grad = 2*reg_strength*W

    return loss, grad

class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)
        
    def grad_zero_(self):
        self.grad = np.zeros_like(self.value)


class ReLULayer:
    def __init__(self):
        self.d_out_x = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
 
        self.d_out_x = X > 0      
        return X * self.d_out_x

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = d_out * self.d_out_x
        
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        a = 1/np.sqrt(n_input)
        self.W = Param(a * np.random.randn(n_input, n_output))
        self.B = Param(a * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X.copy()
        result = X @ self.W.value + self.B.value
        return result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        d_input = d_out @ self.W.value.T
        dW = self.X.T @ d_out
        dB = np.sum(d_out, axis = 0)
        
        self.W.grad += dW
        self.B.grad += dB
        
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
