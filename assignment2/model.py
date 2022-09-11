import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.layers = [
            FullyConnectedLayer(n_input, hidden_layer_size),
            ReLULayer(),
            FullyConnectedLayer(hidden_layer_size, n_output)
            ]


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
    
        for param in self.params().values():
            param.grad_zero_()

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        out = X.copy()
        for layer in self.layers:
            out = layer.forward(out)
            
        loss, d_out = softmax_with_cross_entropy(out, y)
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
            
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        
        if self.reg:
            for param in self.params().values():
                reg_loss, reg_grad = l2_regularization(param.value, self.reg)
                loss += reg_loss
                param.grad += reg_grad
            
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        
        
        pred = np.zeros(X.shape[0], np.int)

        pred = X.copy()
        for layer in self.layers:
            pred = layer.forward(pred)

        pred = np.argmax(pred, axis = 1)
        
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        for idx, layer in enumerate(self.layers):
            for name_param, param in layer.params().items():
                result[name_param + '_' + str(idx)] = param
                
        return result
