import tensorflow as tf
import tensorflow_text as tf_text
import os,sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir));
import config

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

# Loss function ----------------------------------------------------------------------------------
def loss_function(real, pred):
    """
    Calculate the loss function: Currently Categorical Cross entropy. Ignores padding.

    Args:
        real (tf.Tensor): The true labels.
        pred (tf.Tensor): The predicted logits.

    Returns:
        tf.Tensor: The loss value.
    """
    # Currently using sparse categorical crossentropy
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    # Creating mask padding
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    # Calculating loss
    loss = loss_func(real, pred)
    # Casting mask to loss dtype
    mask = tf.cast(mask, dtype=loss.dtype)
    # Multiplying loss by mask
    loss *= mask
    # Sum over non-masked elements normalized by their number
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss

# Accuracy function -------------------------------------------------------------------------------
def accuracy_function(real, pred):
    """
    Calculate the accuracy function. Ignore padding.

    Args:
        real (tf.Tensor): The true labels.
        pred (tf.Tensor): The predicted logits.

    Returns:
        tf.Tensor: The accuracy value.
    """
    # Accuracies calculation
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    # Mask calculation
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    # Applying mask on accuracies
    accuracies = tf.math.logical_and(mask, accuracies)
    # Casting data
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    # Sum over accuracies normalized by non-masked elements
    accuracy = tf.reduce_sum(accuracies) / tf.reduce_sum(mask)
    return accuracy
