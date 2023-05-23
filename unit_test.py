from model.positional_encoding import PositionalEncoder
from model.embedding import TokenEmbedding
from model.pointwise import PointWiseFeedForwardNetwork
import tensorflow as tf
import unittest


class TestPositionalEncoding(unittest.TestCase):
    def __init__(self):
        super().__init__()
        self.embeddings_batch = tf.constant([[[1, 2, 3, 4], [6, 3, 1, 7], [0, 2, 1, 4]], 
                                            [[4, 5, 3, 5],  [5, 4, 3, 6],  [9, 2, 4, 2]]], dtype=tf.float32)
       
    def test_output_shape(self):
        positional_encoder = PositionalEncoder(model_dimension=4, dropout_probability=0.1, max_length=5)
        output = positional_encoder(self.embeddings_batch)
        expected = self.embeddings_batch.shape
        self.assertEqual(output.shape, expected)
        
        
class TestEmbedding(unittest.TestCase):
    def __init__(self):
        super().__init__()
        self.vocab_size = 7
        self.model_dimension = 20
        self.token_ids_batch = tf.constant([[0, 1, 2, 3], 
                                            [1, 3, 4, 5], 
                                            [2, 4, 5, 6]])
        
    def test_output_shape(self):
        embedding = TokenEmbedding(self.vocab_size, self.model_dimension)
        output = embedding(self.token_ids_batch)
        shape = self.token_ids_batch.shape
        batch_size = shape[0]
        sequence_lenght = shape[1]
        self.assertEqual(output.shape, (batch_size, sequence_lenght, self.model_dimension))


class TestPointWise(unittest.TestCase):
    def __init__(self):
        super().__init__()
        self.model_dimension = 4
        # B = 2, S = 3, D = 4
        self.representation_batch = tf.constant([[[1, 2, 3, 4], [6, 3, 1, 7], [0, 2, 1, 4]], 
                                                 [[4, 5, 3, 5],  [5, 4, 3, 6],  [9, 2, 4, 2]]], dtype=tf.float32)  
    def test_output_shape(self):
        pointWiseNet = PointWiseFeedForwardNetwork(self.model_dimension)
        output = pointWiseNet(self.representation_batch)
        expected = self.representation_batch.shape
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

# ALL GOOD
print(' ALL GOOD |'*100)