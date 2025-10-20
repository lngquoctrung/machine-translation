import tensorflow as tf
from tensorflow.keras.losses import Loss

class LabelSmoothingCrossEntropy(Loss):
    """
    Label Smoothing Cross-Entropy

    Formula: y_smooth = (1 - smoothing) * y_true + smoothing / num_classes
    """

    def __init__(self, smoothing=0.1, reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(
            reduction=reduction,
            name="label_smoothing_cross_entropy"
        )
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        """Compute label smoothing cross entropy"""
        vocab_size = tf.cast(tf.shape(y_pred)[-1], tf.float32)

        # Convert to one-hot
        if len(y_true.shape) < len(y_pred.shape):
            y_true = tf.one_hot(
                tf.cast(y_true, tf.int32),
                depth=tf.shape(y_pred)[-1],
            )

        # Apply label smoothing
        y_smooth = y_true * (1 - self.smoothing) + (self.smoothing / vocab_size)

        # Add epsilon for numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Cross entropy
        loss = tf.keras.losses.categorical_crossentropy(
            y_smooth,
            y_pred,
            from_logits=False
        )

        return loss

def get_loss_function(loss_type="label_smoothing", **kwargs):
    """
    Factory function for loss

    Parameters:
        loss_type (string): "label_smoothing" or "sparse_categorical"
    """
    if loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropy(
            smoothing=kwargs.get("smoothing", 0.1),
            reduction="sum_over_batch_size"
        )
    elif loss_type == "sparse_categorical":
        return tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            reduction="sum_over_batch_size"
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
