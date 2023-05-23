import math
import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding

class TokenEmbedding(Layer):
    """
    A layer that creates embeddings for token IDs.

    Attributes:
        vocab_size (int): The size of the vocabulary.
        model_dimension (int): The number of dimensions to use for the embeddings.

    """
    def __init__(self, vocab_size: int, model_dimension: int) -> None:
        """
        Initializes the Embedding layer.

        Args:
            vocab_size (int): The size of the vocabulary.
            model_dimension (int): The number of dimensions to use for the embeddings.

        """
        super().__init__()
        self.model_dimension = model_dimension
        self.embeddings_model = Embedding(vocab_size, model_dimension)
        
    def call(self, token_ids_batch: tf.Tensor) -> tf.Tensor:
        """
        Computes the embeddings of a batch of token IDs.

        Args:
            token_ids_batch (tensorflow.Tensor): A tensor of shape (batch_size, sequence_length) containing token IDs for each token in the batch.

        Returns:
            embeddings (tensorflow.Tensor): A tensor of shape (batch_size, sequence_length, model_dimension) representing the embeddings of each token.

        """
        embeddings = self.embeddings_model(token_ids_batch)
        embeddings *= math.sqrt(self.model_dimension)
        return embeddings

