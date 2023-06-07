import os, sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir));
from model.positional_encoding import PositionalEncoder
from model.embedding import TokenEmbedding
from model.pointwise import PointWiseFeedForwardNetwork
from model.multi_head_attention import MultiHeadAttention
from model.sublayerlogic import SublayerLogic
import tensorflow as tf
import numpy as np
import unittest


class TestPositionalEncoding(unittest.TestCase):
    """Unit tests for the PositionalEncoder class."""

    def __init__(self):
        super().__init__()
        self.embeddings_batch = tf.constant([[[1, 2, 3, 4], [6, 3, 1, 7], [0, 2, 1, 4]], 
                                            [[4, 5, 3, 5],  [5, 4, 3, 6],  [9, 2, 4, 2]]], dtype=tf.float32)
       
    def test_output_shape(self):
        """Test the output shape of the PositionalEncoder."""
        positional_encoder = PositionalEncoder(model_dimension=4, dropout_probability=0.1, max_length=5)
        output = positional_encoder(self.embeddings_batch)
        expected = self.embeddings_batch.shape
        self.assertEqual(output.shape, expected)
        
        
class TestEmbedding(unittest.TestCase):
    """Unit tests for the TokenEmbedding class."""

    def __init__(self):
        super().__init__()
        self.vocab_size = 7
        self.model_dimension = 20
        self.token_ids_batch = tf.constant([[0, 1, 2, 3], 
                                            [1, 3, 4, 5], 
                                            [2, 4, 5, 6]])
        
    def test_output_shape(self):
        """Test the output shape of the TokenEmbedding."""
        embedding = TokenEmbedding(self.vocab_size, self.model_dimension)
        output = embedding(self.token_ids_batch)
        shape = self.token_ids_batch.shape
        batch_size = shape[0]
        sequence_length = shape[1]
        self.assertEqual(output.shape, (batch_size, sequence_length, self.model_dimension))


class TestPointWise(unittest.TestCase):
    """Unit tests for the PointWiseFeedForwardNetwork class."""

    def __init__(self):
        super().__init__()
        self.model_dimension = 4
        # B = 2, S = 3, D = 4
        self.representation_batch = tf.constant([[[1, 2, 3, 4], [6, 3, 1, 7], [0, 2, 1, 4]], 
                                                 [[4, 5, 3, 5],  [5, 4, 3, 6],  [9, 2, 4, 2]]], dtype=tf.float32)  
    def test_output_shape(self):
        """Test the output shape of the PointWiseFeedForwardNetwork."""
        pointWiseNet = PointWiseFeedForwardNetwork(self.model_dimension)
        output = pointWiseNet(self.representation_batch)
        expected = self.representation_batch.shape
        self.assertEqual(output.shape, expected)

class TestMultiHeadAttention(unittest.TestCase):
    """ Unit tests for the MultiHeadAttention class"""
    
    def __init__(self):
        super().__init__()
        B, S= 100, 10
        self.model_dimension = 40
        self.number_heads = 5
        self.vocab_size = 200
        self.mask = None
        token_id_representation_batch = self._generate_synthetic_token_id_representations(B, S, self.vocab_size)
        self.embedding_batch = self._generate_encoding(token_id_representation_batch)
        
    def _generate_synthetic_token_id_representations(self, batch_size, sequence_length, vocab_size):
        """Generates synthetic token id representations """
        return tf.random.uniform(
                                (batch_size, sequence_length),
                                minval=0,
                                maxval=vocab_size-1,
                                dtype=tf.dtypes.float32
                                )
    def _generate_encoding(self, token_id_representations):
        """Generates encoding (Embedding + positional encoding) from token id representations"""
        embedding = TokenEmbedding(self.vocab_size, self.model_dimension)
        x = embedding(token_id_representations)
        positional_encoder = PositionalEncoder(self.model_dimension, dropout_probability=0.1, max_length=10)
        x = positional_encoder(x)
        return x
    
    def test_output_shape(self):
        """Test the output shape of the MultiHeadAttention class"""
        multiHeadAttention = MultiHeadAttention(self.model_dimension, self.number_heads, dropout_probability = 0.1)
        output = multiHeadAttention(query = self.embedding_batch,
                                    key = self.embedding_batch,
                                    value = self.embedding_batch, 
                                    mask = None)
        self.assertEqual(output.shape, self.embedding_batch.shape)

class TestSubLayerLogic(unittest.TestCase):
    """ Unit tests for the sublayer logic """
    
    def __init__(self):
        super().__init__()
        testMHA = TestMultiHeadAttention()
        self.model_dimension = testMHA.model_dimension
        self.embedding_batch = testMHA.embedding_batch
        self.number_heads = testMHA.number_heads 
        multiHeadAttention = MultiHeadAttention(self.model_dimension, self.number_heads, dropout_probability = 0.1)
        self.sublayer_self_attention = lambda representation_batch: multiHeadAttention(query = representation_batch,
                                                                   key = representation_batch,
                                                                   value = representation_batch, 
                                                                   mask = None)
    def test_output_shape(self):
        """ Test the output shape of the TestSubLayerLogic class"""
        sublayerLogic = SublayerLogic(self.model_dimension, dropout=0.1)
        output = sublayerLogic(self.embedding_batch, self.sublayer_self_attention)
        expected = self.embedding_batch.shape
        self.assertEqual(output.shape, expected)
        
# POSITION ENCODING TEST
test_positional_encoding = TestPositionalEncoding()
test_positional_encoding.test_output_shape()

# EMBEDDING TEST
test_embedding = TestEmbedding()
test_embedding.test_output_shape()

# POINTWISE FEED FORWARD NET TEST
test_pointwise = TestPointWise()
test_pointwise.test_output_shape()

# MULTIHEADATTENTION TEST
test_multi_head_attention = TestMultiHeadAttention() 
test_multi_head_attention.test_output_shape()

# SUBLAYERLOGIC (residual + layer norm) TEST
test_sublayer_logic = TestSubLayerLogic()
test_sublayer_logic.test_output_shape()

# ALL GOOD
print(' ALL GOOD |'*100)
