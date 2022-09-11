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


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)
        
    def grad_zero_(self):
        self.grad = np.zeros_like(self.value)
        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        self.d_out_x = X > 0      
        return X * self.d_out_x

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_result = d_out * self.d_out_x
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X.copy()
        result = X @ self.W.value + self.B.value
        return result

    def backward(self, d_out):
        # TODO copy from the previous assignment
        
        d_input = d_out @ self.W.value.T
        dW = self.X.T @ d_out
        dB = np.sum(d_out, axis = 0)
        
        self.W.grad += dW
        self.B.grad += dB
        
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        
        self.X = X.copy()
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        
        if self.padding:
            pad_width = ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0))
            self.X = np.pad(self.X, pad_width, 'constant')
            
        batch_size, height, width, channels = self.X.shape
        
        out_height = height - self.filter_size + 1
        out_width = width - self.filter_size + 1
        out = np.zeros((batch_size, out_height, out_width, self.out_channels))
        
        W = self.W.value.reshape(-1, self.out_channels)
        
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                X_local = self.X[:, x:x+self.filter_size, y:y+self.filter_size, :].reshape(batch_size, -1)
                out[:, x, y, :] = X_local @ W + self.B.value

        return out


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        d_input = np.zeros(self.X.shape)
        self.W.grad = np.zeros(self.W.value.shape)
        self.B.grad = np.zeros(self.B.value.shape)
        W = self.W.value.reshape(-1, self.out_channels)
        
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                
                X_local = self.X[:, x:x+self.filter_size, y:y+self.filter_size, :].reshape(batch_size, -1)
                d_out_local = d_out[:, x, y, :]
                
                d_input_local = (d_out_local @ W.T).reshape(batch_size, self.filter_size, self.filter_size, channels)
                dW_local = (X_local.T @ d_out_local).reshape(self.filter_size, self.filter_size, channels, out_channels)
                dB_local = np.sum(d_out_local, axis = 0)
                
                d_input[:, x:x+self.filter_size, y:y+self.filter_size, :] += d_input_local
                self.W.grad += dW_local
                self.B.grad += dB_local
              
        if self.padding:
            d_input = d_input[:, self.padding:-self.padding, self.padding:-self.padding, :]

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
    
        self.X = X.copy()
        
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        out = np.zeros((batch_size, out_height, out_width, channels))

        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                out[:, x, y, :] = np.max(self.X[:, x*self.stride:x*self.stride+self.pool_size, y*self.stride:y*self.stride+self.pool_size, :], axis = (1,2))

        return out
        
        
    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        d_input = np.zeros(self.X.shape)
        
        for y in range(out_height):
            for x in range(out_width):
                X_local = self.X[:, x*self.stride:x*self.stride+self.pool_size, y*self.stride:y*self.stride+self.pool_size, :]
                mask = np.where(X_local == np.max(X_local, axis=(1,2), keepdims = True), 1, 0)
                
                d_out_local = d_out[:, x, y, :].reshape(batch_size, 1, 1, out_channels)
                d_input[:, x*self.stride:x*self.stride+self.pool_size, y*self.stride:y*self.stride+self.pool_size, :] += mask * d_out_local
                
        return d_input  


    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        self.X_shape = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        batch_size, height, width, channels = self.X_shape
        return d_out.reshape(batch_size, height, width, channels)

    def params(self):
        # No params!
        return {}
