import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense

class DecoderGenerator(Layer):
    """
    Decoder generator connected to last decoder layer. Adds a linear and a softmax over vocabulary (next word prediction)
    
    Attributes:
        linear (Dense): Linear dense layer which will be broadcasted over token in (B, S) set of tokens independantly
    """
    
    def __init__(self, vocab_size):
        self.linear = Dense(vocab_size)

    def call(self, representations_batch):
        """
        Applies generator based on last decoder layer output and gives next token predictions over vocabulary

        Args
            representations_batch (tf.Tensor): The input tensor to the generator. It has shape (B, S, model_dimension)
            
        Returns:
            tf.Tensor: The output tensor of the network of next token prediction for every token. It has shape (B, S, vocab_size).
        """
        return tf.nn.softmax(self.linear(representations_batch), axis=-1)        