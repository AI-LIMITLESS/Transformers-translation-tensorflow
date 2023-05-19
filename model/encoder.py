import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization
from .pointwise import PointWiseFeedForwardNetwork
from .multi_head_attention import MultiHeadedAttention
from .sublayerlogic import SublayerLogic


class Encoder(Layer):
    """
    Transformer encoder composed of multiple encoder layers.

    Attributes:
        encoder_layers (list): List of EncoderLayer instances.
        norm (LayerNormalization): Layer normalization for the encoder output.

    """
    def __init__(self, encoder_layer, number_of_layers):
        """
        Initializes the Transformer encoder.

        Args:
            encoder_layer (EncoderLayer): An instance of the EncoderLayer.
            number_of_layers (int): The number of encoder layers.

        """
        super().__init__()
        assert isinstance(encoder_layer, EncoderLayer), f"First input should be of type EncoderLayer instead of {type(encoder_layer)}"
        self.encoder_layers = [encoder_layer for _ in range(number_of_layers)]
        self.norm = LayerNormalization(encoder_layer.model_dimension)

    def call(self, src_embedding_batch, src_mask):
        """
        Performs the forward pass of the Transformer encoder.

        Args:
            src_embedding_batch (tf.Tensor): A tensor of shape (batch_size, sequence_length, model_dimension) containing source embeddings.
            src_mask (tf.Tensor): A tensor of shape (batch_size, sequence_length) representing the mask for source positions.

        Returns:
            tf.Tensor: A tensor of shape (batch_size, sequence_length, model_dimension) containing the encoded source embeddings.

        """
        src_representation_batch = src_embedding_batch
        for encoder_layer in self.encoder_layers:
            src_representation_batch = encoder_layer(src_representation_batch, src_mask)
        return self.norm(src_representation_batch)


class EncoderLayer(Layer):
    """
    Single layer of the Transformer encoder.

    Attributes:
        sublayers (list): List of SublayerLogic instances.
        multi_head_attention (MultiHeadedAttention): Multi-head attention mechanism.
        pointwise_nn (PointWiseFeedForwardNetwork): Pointwise feed-forward network.
        model_dimension (int): The number of dimensions in the input embeddings.

    """
    def __init__(self, model_dimension, dropout_probability, multi_head_attention, pointwise_nn):
        """
        Initializes the EncoderLayer.

        Args:
            model_dimension (int): The number of dimensions in the input embeddings.
            dropout_probability (float): The probability of dropping out units during training.
            multi_head_attention (MultiHeadedAttention): An instance of MultiHeadedAttention.
            pointwise_nn (PointWiseFeedForwardNetwork): An instance of PointWiseFeedForwardNetwork.

        """
        super().__init__()
        num_of_sublayers_encoder = 2
        self.sublayers = [SublayerLogic(model_dimension, dropout_probability) for _ in range(num_of_sublayers_encoder)]
        self.multi_head_attention = multi_head_attention
        self.pointwise_nn = pointwise_nn
        self.model_dimension = model_dimension

    def call(self, src_representation_batch, src_mask):
        """
        Performs the forward pass of the EncoderLayer.

        Args:
            src_representation_batch (tf.Tensor): A tensor of shape (batch_size, sequence_length, model_dimension) containing source representations.
            src_mask (tf.Tensor): A tensor of shape (batch_size, sequence_length) representing the mask for source positions.

        Returns:
            tf.Tensor: A tensor of shape (batch_size, sequence_length, model_dimension) containing the encoded source representations.

        """
        encoder_self_attention = lambda representations: self.multi_head_attention(query=representations,
                                                                                   key=representations,
                                                                                   value=representations,
                                                                                   mask=src_mask)
        src_representation_batch = self.sublayers[0](src_representation_batch, encoder_self_attention)
        src_representation_batch = self.sublayers[1](src_representation_batch, self.pointwise_nn)
        return src_representation_batch
