from model.positional_encoding import PositionalEncoder
import tensorflow as tf
import unittest


class TestPositionalEncoding(unittest.TestCase):

    def test_output_shape(self):
        embeddings_batch = tf.constant([[[1, 2, 3, 4], [6, 3, 1, 7], [0, 2, 1, 4]], 
                                        [[4, 5, 3, 5], [5, 4, 3, 6], [9, 2, 4, 2]]], dtype=tf.float32)
        positional_encoder = PositionalEncoder(model_dimension=4, dropout_probability=0.1, max_length=5)
        output = positional_encoder(embeddings_batch)
        self.assertEqual(output.shape, (2, 3, 4))
    

test_positional_encoding = TestPositionalEncoding()
test_positional_encoding.test_output_shape()