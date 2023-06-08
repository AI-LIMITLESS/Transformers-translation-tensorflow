import os
import sys
import tensorflow as tf
from data_handler import create_text_vectorizers
from utils_model import create_metrics, create_masks, create_transformer, loss_function
import time
import os,sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir));
import config

class TransformerTrain:
    def __init__(self):
        """
        Initialize the TransformerTrain class.

        """
        # Train step variables signatures
        self.train_step_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64)
        ]
        
        # Text vectorizers
        self.src_text_vectorizer, self.trg_text_vectorizer = create_text_vectorizers()
        
        # Metrics
        self.train_loss, self.train_accuracy = create_metrics()
        
        # Transformer object
        self.transformer = create_transformer()
    
    def train(self, dataset_batches):
        """
        Perform the training process for the transformer model.
           
        Args:
            dataset_batches: BatchDataset
                Dataset batches of training data.
           
        """
        for epoch in range(config.EPOCHS):
            start = time.time()
          
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
          
            # inp -> portuguese, tar -> english
            for (batch, (src, tar)) in enumerate(dataset_batches):
                self.train_step(src, tar)
          
                if batch % 50 == 0:
                    print(f'Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}')
            
    def train_step(self, src, trg):
        """
        Perform a single training step.

        Args:
            src: Tensor, source input sequence.
            trg: Tensor, target input sequence.

        """
        trg_input = trg[:, :-1]
        trg_real = trg[:, 1:]
        src_mask, trg_mask = create_masks(src, trg)
        
        with tf.GradientTape() as tape:
            predictions, _ = self.transformer([src, trg_input], training=True)
            loss = self.loss_function(trg_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(self.accuracy_function(trg_real, predictions))

    def loss_function(self, real, pred):
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