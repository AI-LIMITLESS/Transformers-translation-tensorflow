import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout


class PositionalEncoder(Layer):
    """
    A positional encoding layer for sequences.

    Attributes:
        dropout (Dropout): A dropout layer to apply to the sum of the input embeddings and the positional encodings.
        positional_encodings_table (tf.Variable): A table containing the sinusoidal positional encodings for each position in a sequence.

    """
    def __init__(self, model_dimension: int, dropout_probability: float, max_seq_length: int) -> None:
        """
        Initializes the positional encoding layer.

        Args:
            model_dimension (int): The number of dimensions in the input embeddings.
            dropout_probability (float): The probability of dropping out units during training.
            max_length (int): The maximum length of a sequence to encode.

        """
        super().__init__()
        self.dropout = Dropout(rate=dropout_probability)

        position_id = tf.range(0, max_seq_length, dtype=tf.float32)
        position_id = tf.expand_dims(position_id, axis=1)

        frequencies = (1 / tf.math.pow(10000, tf.range(0, model_dimension, 2, dtype=tf.float32))) / model_dimension

        self.positional_encodings_table = tf.Variable(tf.zeros([max_seq_length, model_dimension]))
        self.positional_encodings_table[:, 0::2].assign(tf.sin(position_id*frequencies))
        self.positional_encodings_table[:, 1::2].assign(tf.cos(position_id*frequencies))

    def call(self, embeddings_batch: tf.Tensor) -> tf.Tensor:
        """
        Computes the positional encodings for a batch of embeddings.

        Args:
            embeddings_batch (tf.Tensor): A tensor of shape (batch_size, sequence_length, model_dimension) containing embeddings to encode.

        Returns:
            tf.Tensor: A tensor of shape (batch_size, sequence_length, model_dimension) containing the encoded embeddings.

        """
        sequence_length = embeddings_batch.shape[1]
        positional_encodings_batch = self.positional_encodings_table[:sequence_length]
        output = self.dropout(embeddings_batch + positional_encodings_batch)
        return output
