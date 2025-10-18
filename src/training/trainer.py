import os
import time
import tensorflow as tf

from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, TensorBoard
)

from .loss_functions import get_loss_function
from .schedulers import get_lr_schedule
from .callbacks import MemoryCallBack
from src.utils.logger import setup_logger
from src.data.dataset import TranslationDataset

class ModelTrainer:
    """
    Trainer with optimization:
        - Label smoothing
        - LR scheduling
        - Mixed precision
        - Memory monitoring
    """
    def __init__(self, model, config, logger_name=__name__, logger_file=None):
        self.model = model
        self.config = config
        self.logger = setup_logger(
            name=logger_name,
            log_file=logger_file
        )

    def compile_model(self):
        """Compile model"""

        # Learning rate schedule
        if self.config.get("use_lr_schedule", True):
            lr_schedule = get_lr_schedule(
                schedule_type="warmup_cosine",
                peak_lr=self.config.get("learning_rate", 0.001),
                warmup_steps=self.config.get("warmup_steps", 4000),
                total_steps=self.config.get("total_steps", 100000),
                min_lr=self.config.get("min_lr", 1e-6)
            )
        else:
            lr_schedule = self.config.get("learning_rate", 0.001)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=self.config.get("beta_1", 0.9),
            beta_2=self.config.get("beta_2", 0.999),
            epsilon=self.config.get("epsilon", 1e-7),
        )

        if self.config.get("use_mixed_precision", False):
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            self.logger.info("Mixed precision optimizer enabled")

        # Loss with label smoothing
        loss_fn = get_loss_function(
            loss_type="label_smoothing",
            smoothing=self.config.get("label_smoothing", 0.1)
        )

        # Compile
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=["accuracy"]
        )

        self.logger.info(f"Model compiled:")
        self.logger.info(f"Loss: Label Smoothing (α={self.config.get('label_smoothing', 0.1)})")
        self.logger.info(f"LR Schedule: {self.config.get('use_lr_schedule', True)}")

    def setup_callbacks(self):
        """Setup callbacks"""
        callbacks = [
            EarlyStopping(
                monitor=self.config.get("monitor", "val_loss"),
                patience=self.config.get("early_stopping_patience", 5),
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(
                    self.config["checkpoint_path"],
                    "best_model.h5"
                ),
                monitor=self.config.get("monitor", "val_loss"),
                save_best_only=self.config.get("save_best_only", True),
                verbose=1
            ),
            TensorBoard(
                log_dir=f"{self.config["log_dir"]}/{time.strftime('%Y%m%d-%H%M%S')}",
                histogram_freq=1,
            ),
            MemoryCallBack(),
        ]
        return callbacks

    def create_tf_dataset(self, data, batch_size, shuffle=True, prefetch=True):
        """Create optimized tf.data.Dataset"""
        src_data, trg_input_data, trg_output_data = data
        translation_dataset = TranslationDataset(
            src_data=src_data,
            trg_input=trg_input_data,
            trg_output=trg_output_data,
            batch_size=batch_size
        )
        dataset = translation_dataset.create_tf_dataset(
            shuffle=shuffle,
            prefetch=prefetch,
        )
        return dataset

    def train(self, train_data, val_data):
        """Train model"""
        self.compile_model()

        batch_size = self.config.get("batch_size", 256)
        train_dataset = self.create_tf_dataset(train_data, batch_size, True, True)
        val_dataset = self.create_tf_dataset(val_data, batch_size, False, False)

        self.logger.info(f"Training started:")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Epochs: {self.config.get('epochs', 10)}")

        start_time = time.time()

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.get("epochs", 10),
            callbacks=self.setup_callbacks(),
            verbose=1,
        )

        elapsed_time = time.time() - start_time
        self.logger.info(f"Training completed in {elapsed_time/3600:.2f} hours")

        return history

    def save_model(self, filepath):
        """Save model"""
        self.model.save(filepath)
        self.logger.info(f"Model saved to {filepath}")
