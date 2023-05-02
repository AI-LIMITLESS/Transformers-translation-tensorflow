import tensorflow as tf

# Redo this part
# class SublayerLogic(tf.keras.layers.Layer):
#     """
#     A residual connection followed by a layer norm.
#     Note for code simplicity the norm is first as opposed to last.
#     """
#     def __init__(self, d_model, dropout):
#         super(SublayerLogic, self).__init__()
#         self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.dropout = tf.keras.layers.Dropout(rate=dropout)

#     def call(self, x, sublayer):
#         # Residual connection between input and sublayer output
#         return x + self.dropout(sublayer(self.norm(x)))