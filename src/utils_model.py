import tensorflow as tf
import tensorflow_text as tf_text
import os,sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir));
import config
from model.transformer import Transformer

# Custom learning rate schedule--------------------------------------------------------------------
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom learning rate schedule."""
    
    def __init__(self, model_dimension, warmup_steps=4000):
        """
        Initialize the CustomSchedule.

        Args:
            model_dimension (int): The dimension of the model.
            warmup_steps (int): The number of warmup steps for the learning rate.
        """
        super().__init__()
        self.model_dimension = model_dimension
        self.model_dimension = tf.cast(self.model_dimension, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        """
        Call method for the CustomSchedule.

        Args:
            step (tf.Tensor): The current step.

        Returns:
            tf.Tensor: The learning rate for the current step.
        """
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** (-1.5))
        return tf.math.rsqrt(self.model_dimension) * tf.math.minimum(arg1, arg2)
    
# Optimizer --------------------------------------------------------------------------------------
def custom_optimizer():
    """
    Create and return an Adam optimizer using Custom LR schedule and constant parameters.

    Returns:
        tf.keras.optimizers.Adam: The Adam optimizer.
    """
    learning_rate = CustomSchedule(model_dimension=config.MODEL_DIMENSION)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=config.BETA_1,
        beta_2=config.BETA_2,
        epsilon=config.EPSILON
    )
    return optimizer

# LOSS + ACCURACY Metrics --------------------------------------------------------------------------
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

# SRC & TARGET masks ----------------------------------------------------------------------------------------
def create_padding_mask(seq):
    """
    Create a padding mask for the input sequence.

    Args:
        seq: Tensor, input sequence.

    Returns:
        Tensor: Padding mask tensor with shape (batch_size, 1, 1, seq_len).

    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(seq_length):
    """
    Create a look-ahead mask for the decoder.

    Args:
        seq_length: int, length of the sequence.

    Returns:
        Tensor: Look-ahead mask tensor with shape (seq_len, seq_len).

    """
    mask = 1 - tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)
    return mask


def create_masks(src, trg):
    """
    Create masks for the encoder and decoder inputs.

    Args:
        src: Tensor, source input sequence.
        trg: Tensor, target input sequence.

    Returns:
        Tuple[Tensor, Tensor]: Source mask and target mask.

    """
    # Encoder padding mask
    src_mask = create_padding_mask(src)
    # Decoder padding mask
    trg_padding_mask = create_padding_mask(trg)
    # Decoder look ahead mask
    trg_look_ahead_mask = create_look_ahead_mask(tf.shape(trg)[1])
    # Decoder mask (joining padding + lookahead)
    trg_mask = tf.maximum(trg_padding_mask, trg_look_ahead_mask)
    return src_mask, trg_mask

# Transformer instantiation ------------------------------------------------------------------------
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

