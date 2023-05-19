import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, Dropout

class SublayerLogic(Layer):
    """
    A residual connection preceeded by a layer norm.

    Attributes:
        model_dimension (int): The expected dimensionality of the input tensor.
        dropout (float): The rate of dropout to apply.
        layer_norm (LayerNormalization): A layer normalization layer.
        dropout_layer (Dropout): A dropout layer.

    """

    def __init__(self, model_dimension: int, dropout: float):
        """
        Initializes the SublayerLogic layer.

        Args:
            model_dimension (int): The expected dimensionality of the input tensor.
            dropout (float): The rate of dropout to apply.
        """
        super().__init__()
        self.model_dimension = model_dimension
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        self.dropout_layer = Dropout(rate=dropout)

    def call(self, representations_batch, sub_layer):
        """
        Passes the input tensor through the sub-layer and adds the original input tensor.

        Args:
            representations_batch (tf.Tensor): A tensor to pass through the sub-layer.
            sub_layer (Layer): A layer to apply to the input tensor.
            
        Returns:
            tf.Tensor: The result of the sub-layer applied to the input tensor added to the input tensor.
        """
        return representations_batch + self.dropout_layer(sub_layer(self.layer_norm(representations_batch)))
