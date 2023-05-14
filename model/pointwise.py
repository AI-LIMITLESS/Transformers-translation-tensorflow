import tensorflow as tf
from tensorflow.keras.layers import Layer
from keras.layers import Dense, Dropout


class PointWiseFeedForwardNetwork(Layer):
    """
    A fully connected feed-forward network, which is applied to each token separately and identically.
    The broadcast on the first two dimensions (B, S) is done automatically by TensorFlow.
    The layers consist of two linear transformations with a ReLU activation in between (autoencoder).
    
    Attributes:
        layers (list): List of layers representating an auto-encoder with dropout in the middle
    """
    
    def __init__(self, model_dimension, width_mul=4, dropout=0.1):
        """
        Initializes the PointWiseFeedForwardNetwork.
        
        Args:
            model_dimension (int): The number of output dimensions of each dense layer.
            width_mul (int, optional): The multiplier to determine the number of hidden units in the first dense layer.
            dropout (float, optional): The rate of dropout regularization applied to the output of the first dense layer.
        """
        super().__init__()
        self.layers = [
             Dense(model_dimension * width_mul, activation='relu', input_shape=model_dimension),
             Dropout(dropout),
             Dense(model_dimension)
         ]
    
    def call(self, representations_batch):
        """
        Runs the forward pass of the PointWiseFeedForwardNetwork.
        
        Args:
            representations_batch (tf.Tensor): The input tensor to the network. It has shape (B, S, model_dimension).
            
        Returns:
            tf.Tensor: The output tensor of the network. It has shape (B, S, model_dimension).
        """
        return self.layers(representations_batch)
