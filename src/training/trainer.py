import os
import time
import pickle
import re
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard
)
from .loss_functions import get_loss_function
from .schedulers import get_lr_schedule
from .callbacks import MemoryCallBack, SaveHistoryCallback, PeriodicCheckpoint
from src.utils import setup_logger, sanitize_path
from src.data import TranslationDataset

class ModelTrainer:
    """
    Trainer with optimization:
    - Label smoothing
    - LR scheduling
    - Mixed precision
    - Memory monitoring
    - Resume training support
    """
    
    def __init__(self, model, config, model_name=None, logger_name=__name__, logger_file=None):
        self.model = model
        self.config = config
        self.model_name = model_name
        self.logger = setup_logger(
            name=logger_name,
            log_file=logger_file
        )
        self.initial_epoch = 0
    
    def compile_model(self):
        """Compile model"""
        # Learning rate schedule
        if self.config.USE_LR_SCHEDULER:
            lr_schedule = get_lr_schedule(
                schedule_type="warmup_cosine",
                peak_lr=self.config.LEARNING_RATE,
                warmup_steps=self.config.WARMUP_STEPS,
                total_steps=self.config.TOTAL_STEPS,
                min_lr=self.config.MIN_LR
            )
        else:
            lr_schedule = self.config.LEARNING_RATE
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=self.config.BETA_1,
            beta_2=self.config.BETA_2,
            epsilon=self.config.OPTIMIZER_EPSILON,
        )
        
        if self.config.USE_MIXED_PRECISION:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                optimizer,
                dynamic=self.config.OPTIMIZER_DYNAMIC,
                initial_scale=self.config.OPTIMIZER_INITIAL_SCALE,
                dynamic_growth_steps=self.config.OPTIMIZER_DYNAMIC_GROWTH_STEPS
            )
            self.logger.info("Mixed precision optimizer enabled with increased stability")
        
        # Loss with label smoothing
        loss_fn = get_loss_function(
            loss_type="label_smoothing",
            smoothing=self.config.LABEL_SMOOTHING
        )
        
        # Compile
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=["accuracy"],
        )
        
        self.logger.info(f"Model compiled:")
        self.logger.info(f"Loss: Label Smoothing (α={self.config.LABEL_SMOOTHING})")
        self.logger.info(f"LR Schedule: {self.config.USE_LR_SCHEDULER}")
    
    def load_checkpoint(self, checkpoint_path=None):
        """Load checkpoint to resume training"""
        if checkpoint_path is None:
            # Automaticallyy find latest checkpoint
            checkpoint_dir = os.path.join(
                self.config.ARTIFACT_PATH, 
                self.model_name,
                "checkpoints"
            )
            if not os.path.exists(checkpoint_dir):
                self.logger.info("No checkpoint directory found. Starting from scratch.")
                return
            # Get filename of checkpoints in directory
            checkpoints = [f for f in os.listdir(checkpoint_dir) 
                          if f.startswith("checkpoint_epoch_") and f.endswith(".h5")]
            
            if not checkpoints:
                self.logger.info("No checkpoint found. Starting from scratch.")
                return
            
            # Sort and get latest checkpoint
            checkpoints.sort()
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
        
        if os.path.exists(checkpoint_path):
            self.logger.info(f"Loading checkpoint: {checkpoint_path}")
            self.model = tf.keras.models.load_model(checkpoint_path)
            
            # Get epoch number from filename
            match = re.search(r'epoch_(\d+)', checkpoint_path)
            if match:
                self.initial_epoch = int(match.group(1))
                self.logger.info(f"Resuming from epoch {self.initial_epoch}")
        else:
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
    
    def setup_callbacks(self):
        """Setup callbacks"""
        history_path = os.path.join(
            self.config.ARTIFACT_PATH,
            self.model_name, 
            "training_history.pkl"
        )
        checkpoint_path = os.path.join(
            self.config.ARTIFACT_PATH, 
            self.model_name,
            "checkpoints"
        )
        
        callbacks = [
            tf.keras.callbacks.TerminateOnNaN(),
            EarlyStopping(
                monitor=self.config.MONITOR,
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            # Best model checkpoint
            ModelCheckpoint(
                filepath=os.path.join(
                    checkpoint_path,
                    "best_model.h5"
                ),
                monitor=self.config.MONITOR,
                save_best_only=self.config.SAVE_BEST_ONLY,
                verbose=1
            ),
            PeriodicCheckpoint(
                filepath=os.path.join(
                    checkpoint_path,
                    "checkpoint_epoch_{epoch:03d}.h5"
                ),
                save_every_n_epochs=10
            ),
            TensorBoard(
                log_dir=f"{self.config.LOG_DIR}/{time.strftime('%Y%m%d-%H%M%S')}",
                histogram_freq=1,
            ),
            MemoryCallBack(),
            SaveHistoryCallback(filepath=history_path),
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
    
    def train(self, train_data, val_data, resume=False):
        """Train model with resume capability"""
        if resume:
            self.load_checkpoint()
        
        self.compile_model()
        batch_size = self.config.BATCH_SIZE
        
        # Create datasets
        train_dataset = self.create_tf_dataset(train_data, batch_size, True, True)
        val_dataset = self.create_tf_dataset(val_data, batch_size, False, False)
        
        total_epochs = self.config.EPOCHS
        
        self.logger.info(f"Training started:")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Initial epoch: {self.initial_epoch}")
        self.logger.info(f"Total epochs: {total_epochs}")
        
        start_time = time.time()
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=total_epochs,
            initial_epoch=self.initial_epoch,
            callbacks=self.setup_callbacks(),
            verbose=1,
        )
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Training completed in {elapsed_time/3600:.2f} hours")
        
        history_path = os.path.join(
            self.config.ARTIFACT_PATH,
            self.model_name, 
            "training_history.pkl"
        )
        
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                complete_history = pickle.load(f)
            
            class CombinedHistory:
                def __init__(self, history_dict):
                    self.history = history_dict
            
            return CombinedHistory(complete_history)
        
        return history
    
    def save_model(self, filepath):
        """Save model"""
        self.model.save(filepath)
        self.logger.info(f"Model saved to {sanitize_path(filepath)}")
