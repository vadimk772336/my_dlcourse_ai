import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        _, _, n_channels = input_shape
        
        self.layers = [
            ConvolutionalLayer(n_channels, conv1_channels, 3, 1),       #32х32xconv1_channels
            ReLULayer(),
            MaxPoolingLayer(4, 4),                                      #8х8xconv1_channels
            ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1),   #8х8xconv2_channels
            ReLULayer(),
            MaxPoolingLayer(4, 4),                                      #2х2xconv2_channels
            Flattener(), 
            FullyConnectedLayer(conv2_channels*2*2, n_output_classes)   #conv2_channels*2*2
            ]
        
        

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        for param in self.params().values():
            param.grad_zero_()
            
        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        
        out = X.copy()
        for layer in self.layers:
            out = layer.forward(out)
            
        loss, d_out = softmax_with_cross_entropy(out, y)
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
            
        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment

        pred = X.copy()
        for layer in self.layers:
            pred = layer.forward(pred)

        pred = np.argmax(pred, axis = 1)
        
        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        
        for idx, layer in enumerate(self.layers):
            for name_param, param in layer.params().items():
                result[name_param + '_' + str(idx)] = param
                
        return result
