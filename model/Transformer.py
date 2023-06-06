from .embedding import TokenEmbedding 
from .positional_encoding import PositionalEncoding
from .sublayerlogic import SublayerLogic
from .pointwise import PointWiseFeedForwardNetwork
from .multi_head_attention import MultiHeadedAttention
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .decoder_generator import DecoderGenerator
import tensorflow as tf
from tensorflow.keras.layers import Layer

class Transformer(Layer):
    """
    The Transformer model consisting of an encoder and decoder.

    Attributes:
        src_embedding (TokenEmbedding): The embedding layer for the source token IDs.
        trg_embedding (TokenEmbedding): The embedding layer for the target token IDs.
        src_pos_encoding (PositionalEncoding): The positional encoding layer for the source token IDs.
        trg_pos_encoding (PositionalEncoding): The positional encoding layer for the target token IDs.
        encoder (Encoder): The encoder component of the Transformer model.
        decoder (Decoder): The decoder component of the Transformer model.
        decoder_generator (DecoderGenerator): The decoder generator for generating output tokens.

    """
    def __init__(
        self,
        model_dimension: int,
        src_vocab_size: int,
        trg_vocab_size: int,
        number_heads: int,
        number_layers: int,
        dropout_probability: float,
        max_seq_length: int = 500,
    ) -> None:
        """
        Initializes the Transformer model.

        Args:
            model_dimension (int): The number of dimensions for the token embeddings and positional encodings.
            src_vocab_size (int): The size of the source vocabulary.
            trg_vocab_size (int): The size of the target vocabulary.
            number_heads (int): The number of attention heads.
            number_layers (int): The number of layers in the encoder and decoder.
            dropout_probability (float): The probability of dropping out units during training.
            max_seq_length (int, optional): The maximum length of a sequence. Defaults to 500.

        """
        super().__init__()

        # 1. Embedding src/trg token IDs into vectors
        self.src_embedding = TokenEmbedding(src_vocab_size, model_dimension)
        self.trg_embedding = TokenEmbedding(trg_vocab_size, model_dimension)

        # 2. Positional encoding of src/trg token IDs
        self.src_pos_encoding = PositionalEncoding(model_dimension, dropout_probability, max_seq_length)
        self.trg_pos_encoding = PositionalEncoding(model_dimension, dropout_probability, max_seq_length)

        # 3. Sublayers: Multi-head-attention & PositionWise net
        mha = MultiHeadedAttention(model_dimension, number_heads, dropout_probability)
        pwn = PointWiseFeedForwardNetwork(model_dimension, dropout=dropout_probability)

        # 4. Encoder/Decoder layer
        encoder_layer = EncoderLayer(model_dimension, dropout_probability, mha, pwn)
        decoder_layer = DecoderLayer(model_dimension, dropout_probability, mha, pwn)

        # 5. Encoder/Decoder
        self.encoder = Encoder(encoder_layer, number_layers)
        self.decoder = Decoder(decoder_layer, number_layers)

        # 6. Decoder Generator
        self.decoder_generator = DecoderGenerator(model_dimension, trg_vocab_size)

    def call(self, src_token_ids_batch, trg_token_ids_batch, src_mask, trg_mask):
        """
        Computes transformer forward pass

        Args:
            src_token_ids_batch (tf.tensor): A tensor of shape (batch_size, sequence_length) containing token IDs for each token in the src batch.
            trg_token_ids_batch (tf.tensor): A tensor of shape (batch_size, sequence_length) containing token IDs for each token in the trg batch.
            src_mask (tf.tensor): src input mask tensor with shape (batch_size, number_heads, sequence_length) where False indicates that the corresponding element should be masked out.
            trg_mask (tf.tensor): trg input mask tensor with shape (batch_size, number_heads, sequence_length)
        
        Returns:
            tf.tensor: Tensor of size (B, S, V) representing transformer output (log probs through vocab)
            
        """
        src_representations_batch = self.encode(src_token_ids_batch, src_mask)
        trg_log_probs = self.decode(src_representations_batch, trg_token_ids_batch, src_mask, trg_mask)
        return trg_log_probs
    
    def encode(self, src_token_ids_batch, src_mask):
        """
        Encodes src token ids by using: Token embedding, positional encoding, encoder layers

        Args:
            src_token_ids_batch (tf.tensor): A tensor of shape (batch_size, sequence_length) containing token IDs for each token in the src batch.
            src_mask (tf.tensor): src input mask tensor with shape (batch_size, number_heads, sequence_length) where False indicates that the corresponding element should be masked out.
        
        Returns:
            tf.tensor: Tensor of size (B, S, Model dimension) representing encoder output
        """
        # Token embedding
        src_embeddings_batch = self.src_embedding(src_token_ids_batch)
        # Token positional encoding
        src_embeddings_batch = self.src_pos_encoding(src_embeddings_batch)
        # Encoder forward pass
        src_representations_batch = self.encoder(src_embeddings_batch, src_mask)
        return src_representations_batch
    
    def decode(self, src_representations_batch, trg_token_ids_batch, src_mask, trg_mask):
        """
        Forward pass through decoder & decoder generator after target token embedding and positional encoding
        
        Args:
            src_token_ids_batch (tf.tensor): A tensor of shape (batch_size, sequence_length) containing token IDs for each token in the src batch.
            trg_token_ids_batch (tf.tensor): A tensor of shape (batch_size, sequence_length) containing token IDs for each token in the trg batch.
            src_mask (tf.tensor): src input mask tensor with shape (batch_size, number_heads, sequence_length) where False indicates that the corresponding element should be masked out.
            trg_mask (tf.tensor): trg input mask tensor with shape (batch_size, number_heads, sequence_length)
        
        Returns:
            tf.tensor: Tensor of size (B*S, Target Vocabulary size) representing decoder generator output
        """
        # Token embedding
        trg_embeddings_batch = self.trg_embedding(trg_token_ids_batch)
        # Token positional encoding
        trg_embeddings_batch = self.trg_pos_encoding(trg_embeddings_batch)
        # Decoder forward pass
        trg_representations_batch = self.decoder(src_representations_batch, trg_embeddings_batch, src_mask, trg_mask)
        # Decoder generator - output shape: (B, S, trg_vocab_size)
        trg_log_probs = self.decoder_generator(trg_representations_batch)
        # Reshaping decoder generator into (B*S, trg_vocab_size) to facilitate the use of KL divergence loss function during training
        trg_log_probs = trg_log_probs.reshape(-1, trg_log_probs.shape[-1])
        return trg_log_probs