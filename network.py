import numpy as np
import data
import time
import tqdm

"""
NOTE
----
Start by implementing your methods in non-vectorized format - use loops and other basic programming constructs.
Once you're sure everything works, use NumPy's vector operations (dot products, etc.) to speed up your network.
"""

def sigmoid(a):
    return 1/ (1 + np.exp(-a))
    """
    Compute the sigmoid function.

    f(x) = 1 / (1 + e ^ (-x))

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying sigmoid (z from the slides).
    """

def softmax(a):
    exponent = np.exp(a)
    return np.divide(exponent.T,np.sum(exponent, axis=1).T).T
    """
    Compute the softmax function.

    f(x) = (e^x) / Σ (e^x)

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying softmax (z from the slides).
    """

def binary_cross_entropy(y, t):
    return -1*(t.T @ np.log(y) + (1-t).T @ np.log(1-y))
    """
    Compute binary cross entropy.

    L(x) = t*ln(y) + (1-t)*ln(1-y)

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        binary cross entropy loss value according to above definition
    """

def multiclass_cross_entropy(y, t):
    return -1*np.einsum("ij,ij",np.log(y),t)

    """
    Compute multiclass cross entropy.

    L(x) = - Σ (t*ln(y))

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        multiclass cross entropy loss value according to above definition
    """

class Network:
    def __init__(self, hyperparameters, activation, loss, out_dim):
        """
        Perform required setup for the network.

        Initialize the weight matrix, set the activation function, save hyperparameters.

        You may want to create arrays to save the loss values during training.

        Parameters
        ----------
        hyperparameters
            A Namespace object from `argparse` containing the hyperparameters
        activation
            The non-linear activation function to use for the network
        loss
            The loss function to use while training and testing
        """
        self.hyperparameters = hyperparameters
        self.learning_rate = hyperparameters.learning_rate
        self.activation = activation
        self.loss = loss

        self.weights = np.zeros((32*32+1, out_dim))

    def forward(self, X):
        return self.activation(X@self.weights)
        """
        Apply the model to the given patterns

        Use `self.weights` and `self.activation` to compute the network's output

        f(x) = σ(w*x)
            where
                σ = non-linear activation function
                w = weight matrix

        Make sure you are using matrix multiplication when you vectorize your code!

        Parameters
        ----------
        X
            Patterns to create outputs for
        """
    def __call__(self, X):
        return self.forward(X) # retunrn this when the network class instance is created

    def train(self,minibatch):
        X, t = minibatch # Takes in a single minibatch
        # a = self.hyperparameters[1] # Get step size

        batch_size = np.shape(t)[0] # Batch Size of average normalization
        y = self.forward(X) # avtivation fn
        self.weights = self.weights + self.learning_rate / batch_size * ((t - y).T @ X).T
        average_loss = self.loss(y=self.forward(X),t=t)/batch_size
        if(self.activation == softmax):
                accuracy = np.sum(np.argmax(self.forward(X),axis = 1) == data.onehot_decode(t))/np.shape(t)[0]
        else:
            accuracy = np.sum(np.greater(self.forward(X),0.5) == t)/np.shape(t)[0]
        return(average_loss, accuracy)

    def test(self, minibatch):
        (X, t) = minibatch
        batch_size = np.shape(t)[0] # Batch Size of average normalization
        average_loss = self.loss(y=self.forward(X),t=t)/batch_size
        if(self.activation == softmax):
            accuracy = np.sum(np.argmax(self.forward(X),axis = 1) == data.onehot_decode(t))/np.shape(t)[0]
        else:
            accuracy = np.sum(np.greater(self.forward(X),0.5) == t)/np.shape(t)[0]
        return(average_loss, accuracy)

        """
        Test the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` to compute the loss.
        Do NOT update the weights in this method!

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
            tuple containing:
                average loss over minibatch
                accuracy over minibatch
        """