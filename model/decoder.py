import tensorflow as tf
import copy
from tensorflow.keras.layers import Layer, LayerNormalization
from .pointwise import PointWiseFeedForwardNetwork
from .multi_head_attention import MultiHeadedAttention
from .sublayerlogic import SublayerLogic


class Decoder(Layer):
    """
    Transformer decoder composed of multiple decoder layers.

    Attributes:
        decoder_layers (list): List of DecoderLayer instances.
        norm (LayerNormalization): Layer normalization for the decoder output.

    """
    def __init__(self, decoder_layer, number_of_layers):
        """
        Initializes the Transformer decoder.

        Args:
            decoder_layer (DecoderLayer): An instance of the DecoderLayer.
            number_of_layers (int): The number of decoder layers.

        """
        super().__init__()
        assert isinstance(decoder_layer, DecoderLayer), f"First input should be of type DecoderLayer instead of {type(decoder_layer)}"
        self.decoder_layers = [copy.deepcopy(decoder_layer) for _ in range(number_of_layers)]
        self.norm = LayerNormalization(decoder_layer.model_dimension)

    def call(self, trg_embedding_batch, src_representation_batch, trg_mask, src_mask):
        """
        Performs the forward pass of the Transformer decoder.

        Args:
            trg_embedding_batch (tf.Tensor): A tensor of shape (batch_size, sequence_length, model_dimension) containing target embeddings.
            src_representation_batch (tf.Tensor): A tensor of shape (batch_size, source_length, model_dimension) containing source representations.
            trg_mask (tf.Tensor): A tensor of shape (batch_size, sequence_length) representing the mask for target positions.
            src_mask (tf.Tensor): A tensor of shape (batch_size, source_length) representing the mask for source positions.

        Returns:
            tf.Tensor: A tensor of shape (batch_size, sequence_length, model_dimension) containing the encoded target representations.

        """
        trg_representation_batch = trg_embedding_batch
        for decoder_layer in self.decoder_layers:
            trg_representation_batch = decoder_layer(trg_representation_batch, src_representation_batch, trg_mask, src_mask)
        return self.norm(trg_representation_batch)


class DecoderLayer(Layer):
    """
    Single layer of the Transformer decoder.

    Attributes:
        sublayers (list): List of SublayerLogic instances.
        trg_multi_head_attention (MultiHeadedAttention): Multi-head attention mechanism for target positions.
        src_multi_head_attention (MultiHeadedAttention): Multi-head attention mechanism for source positions.
        pointwise_nn (PointWiseFeedForwardNetwork): Pointwise feed-forward network.
        model_dimension (int): The number of dimensions in the input embeddings.

    """
    def __init__(self, model_dimension, dropout_probability, multi_head_attention, pointwise_nn):
        """
        Initializes the DecoderLayer.

        Args:
            model_dimension (int): The number of dimensions in the input embeddings.
            dropout_probability (float): The probability of dropping out units during training.
            multi_head_attention (MultiHeadedAttention): An instance of MultiHeadedAttention.
            pointwise_nn (PointWiseFeedForwardNetwork): An instance of PointWiseFeedForwardNetwork.

        """
        super().__init__()
        num_of_sublayers_encoder = 3
        self.sublayers = [SublayerLogic(model_dimension, dropout_probability) for _ in range(num_of_sublayers_encoder)]
        self.trg_multi_head_attention = copy.deepcopy(multi_head_attention)
        self.src_multi_head_attention = copy.deepcopy(multi_head_attention)
        self.pointwise_nn = pointwise_nn
        self.model_dimension = model_dimension

    def call(self, src_representation_batch, trg_representation_batch, src_mask, trg_mask):
        """
        Performs the forward pass of the DecoderLayer.

        Args:
            trg_representation_batch (tf.Tensor): A tensor of shape (batch_size, sequence_length, model_dimension) containing target representations.
            src_representation_batch (tf.Tensor): A tensor of shape (batch_size, source_length, model_dimension) containing source representations.
            trg_mask (tf.Tensor): A tensor of shape (batch_size, sequence_length) representing the mask for target positions.
            src_mask (tf.Tensor): A tensor of shape (batch_size, source_length) representing the mask for source positions.

        Returns:
            tf.Tensor: A tensor of shape (batch_size, sequence_length, model_dimension) containing the encoded target representations.

        """
        srb = src_representation_batch
        decoder_trg_self_attention = lambda trb: self.trg_multi_headed_attention(query=trb, key=trb, value=trb, mask=trg_mask)
        decoder_src_attention = lambda trb: self.src_multi_headed_attention(query=trb, key=srb, value=srb, mask=src_mask)
        trg_representation_batch = self.sublayers[0](trg_representation_batch, decoder_trg_self_attention)
        trg_representation_batch = self.sublayers[1](trg_representation_batch, decoder_src_attention)
        trg_representation_batch = self.sublayers[2](trg_representation_batch, self.pointwise_net)

        return trg_representation_batch
