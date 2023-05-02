import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout


class PointWiseFeedForwardNetwork(Sequential):
    """
    A fully connected feed-forward network, which is applied to each position separately and identically.
    This consists of two linear transformations with a ReLU activation in between.
    The first layer takes inputs in model dimension and outputs in model dimension * width_ffn.
    and vice versa for the second layer
    """
    def __init__(self, d_model, dff, dropout=0.1):
        super(PointWiseFeedForwardNetwork, self).__init__(layers)
        
        layers = [
            Dense(dff, activation='relu'),
            Dropout(dropout),
            Dense(d_model)
        ]

        