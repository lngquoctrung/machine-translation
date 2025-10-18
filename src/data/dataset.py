import tensorflow as tf
import numpy as np

class TranslationDataset:
    def __init__(self, src_data, trg_input, trg_output, batch_size=128):
        self.src_data = src_data
        self.trg_input = trg_input
        self.trg_output = trg_output
        self.batch_size = batch_size
        self.num_samples = len(self.src_data)

    def create_tf_dataset(self, shuffle=True, prefetch=True):
        """Create optimized tf.data.Dataset"""
        dataset = tf.data.Dataset.from_tensor_slices((
            (self.src_data, self.trg_input),
            self.trg_output,
        ))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(10000, self.num_samples))

        dataset = dataset.batch(self.batch_size)

        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))