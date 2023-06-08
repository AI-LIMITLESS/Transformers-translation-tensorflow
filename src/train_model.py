import os, sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir));
import config
from model.transformer import Transformer
import tensorflow as tf

def create_transformer() -> Transformer:
    """
    Creates a transformer instance with config parameters

    Returns
    -------
    Transformer
        return a Transformer object

    """
    transformer = Transformer(model_dimension = config.MODEL_DIMENSION,
                              src_vocab_size = config.VOCAB_SIZE,
                              trg_vocab_size = config.VOCAB_SIZE,
                              number_heads = config.NUM_HEADS,
                              number_layers = config.NUM_LAYERS,
                              dropout_probability = config.DROPOUT_PROBABILITY,
                              max_seq_length = config.MAX_SEQ_LENGTH)
    return transformer

def create_metrics() -> tuple:
    """
    Create and generate train loss and accuracy metrics

    Returns
    -------
    tuple
        A tuple of keras metrics (mean) of train loss & accuracy

    """
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    return train_loss, train_accuracy