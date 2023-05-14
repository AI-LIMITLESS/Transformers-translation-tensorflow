import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout

class MultiHeadedAttention(Layer):
    """
    A multi-headed attention layer for transformer models.
    
    Attributes:
        head_dimension (int): The dimension of each attention head.
        number_heads (int): The number of attention heads.
        qkv_nets (list): List of three dense layers for computing query, key, and value projections.
        out_projection_net (Dense): Dense layer for projecting the output of the attention layer.
        attention_dropout (Dropout): Dropout layer applied to the attention weights.
    """
    
    def __init__(self, model_dimension, number_heads, dropout_probability):
        """
        Initializes the MultiHeadedAttention layer.
        
        Args:
            model_dimension (int): The dimension of the input tokens.
            number_heads (int): The number of attention heads.
            dropout_probability (float): Dropout probability applied to the attention weights.
        """
        super().__init__()
        assert model_dimension % number_heads == 0, f'Model dimension must be divisible by the number of heads.'
        
        self.head_dimension = int(model_dimension / number_heads)
        self.number_heads = number_heads
        
        self.qkv_nets = [Dense(model_dimension, model_dimension) for _ in range(3)]
        self.out_projection_net = Dense(model_dimension, model_dimension)
       
        self.attention_dropout = Dropout(rate = dropout_probability)
        
    def attention(self, query, key, value, mask=None):
        """
        Performs the scaled dot-product attention operation.
    
        Args:
            query (tf.Tensor): Query tensor with shape (batch_size, number_heads, sequence_length, head_dimension).
            key (tf.Tensor): Key tensor with shape (batch_size, number_heads, sequence_length, head_dimension).
            value (tf.Tensor): Value tensor with shape (batch_size, number_heads, sequence_length, head_dimension).
            mask (tf.Tensor, optional): Mask tensor with shape (batch_size, number_heads, sequence_length) where 
                False indicates that the corresponding element should be masked out.
    
        Returns:
            tf.Tensor: Tensor with shape (batch_size, number_heads, sequence_length, head_dimension) 
                       representing the attention output.
        """
        scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(self.head_dimension)
        if mask is not None:
            scores = tf.where(mask == tf.constant(False), scores, float('-inf'))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = self.attention_dropout(attention_weights)
        token_representations = tf.matmul(attention_weights, value)
        return token_representations
    
    def forward(self, query, key, value, mask):
        """
        Performs the forward pass for the multi-headed attention layer.

        Args:
            query (tf.Tensor): A tensor of shape (batch_size, sequence_length, model_dimension), representing the input query.
            key (tf.Tensor): A tensor of shape (batch_size, sequence_length, model_dimension), representing the input key.
            value (tf.Tensor): A tensor of shape (batch_size, sequence_length, model_dimension), representing the input value.
            mask (tf.Tensor, optional): A tensor of shape (batch_size, sequence_length), representing the mask to be applied to the input.
        
        Returns:
            tf.Tensor: A tensor of shape (batch_size, sequence_length, model_dimension), representing the output of the multi-headed attention layer.
        """
        # Linear + Reshaping query, key, value to multi-heads 
        query, key, value = [net(x) for net, x in zip(self.qkv_nets, (query, key, value))]
        query, key, value = [tf.reshape(x, [x.shape[0], -1, self.number_heads, self.head_dimension]) for x in (query, key, value)]
        query, key, value = [tf.transpose(x, perm=[0, 2, 1, 3]) for x in (query, key, value)]
        # Attention values
        token_representations = self.attention(query, key, value, mask)
        # Reshape back to origin
        token_representations = tf.reshape(tf.transpose(token_representations, perm=[0, 2, 1, 3]), [token_representations.shape[0], -1, self.number_heads*self.head_dimension])
        return token_representations
