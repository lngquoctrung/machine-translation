import time
import pickle
import os
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from src.utils import setup_logger

class MemoryCallBack(Callback):
    """Monitor GPU memory"""
    def __init__(self, name_logger=__name__, filename_logger=None):
        super().__init__()
        self.logger = setup_logger(
            name=name_logger,
            log_file=filename_logger
        )
        self.peak_memory = 0
    
    def on_epoch_end(self, epoch, logs=None):
        """Print memory after each epoch"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                memory_used = gpu.memoryUsed
                memory_total = gpu.memoryTotal
                memory_percent = memory_used / memory_total * 100
                self.peak_memory = max(self.peak_memory, memory_used)
                self.logger.info(
                    f"GPU Memory: {memory_used:.0f}/{memory_total:.0f} MB "
                    f"({memory_percent:.1f}%) | Peak: {self.peak_memory:.0f} MB"
                )
        except Exception as e:
            self.logger.warning(f"Could not read GPU memory: {e}")

class TimeHistoryCallBack(Callback):
    """Track training time"""
    def __init__(self, name_logger=__name__, filename_logger=None):
        super().__init__()
        self.times = []
        self.logger = setup_logger(
            name=name_logger,
            log_file=filename_logger,
        )
        self.epoch_time_start = None
    
    def on_epoch_begin(self, epoch, logs=None):
        """Start timer at epoch begin"""
        self.epoch_time_start = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        """Log time at epoch end"""
        epoch_time = time.time() - self.epoch_time_start
        self.times.append(epoch_time)
        avg_time = sum(self.times) / len(self.times)
        self.logger.info(
            f"Epoch {epoch + 1} | Time: {epoch_time:.2f}s | Average: {avg_time:.2f}s"
        )

class SaveHistoryCallback(Callback):
    """Save training history to pickle file after each epoch"""
    
    def __init__(self, filepath, name_logger=__name__):
        super().__init__()
        self.filepath = filepath
        self.logger = setup_logger(name=name_logger)
        self.history = {}
        
        # Load existing history if resuming
        if os.path.exists(self.filepath):
            with open(self.filepath, 'rb') as f:
                self.history = pickle.load(f)
                self.logger.info(f"Loaded existing history with {len(self.history.get('loss', []))} epochs")
    
    def on_epoch_end(self, epoch, logs=None):
        """Append metrics to history and save"""
        logs = logs or {}
        
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(float(value))
        
        # Save to file after each epoch
        with open(self.filepath, 'wb') as f:
            pickle.dump(self.history, f)

class PeriodicCheckpoint(Callback):
    """Save checkpoint every N epochs"""
    
    def __init__(self, filepath, save_every_n_epochs=5, name_logger=__name__):
        super().__init__()
        self.filepath = filepath
        self.save_every_n_epochs = save_every_n_epochs
        self.logger = setup_logger(name=name_logger)
    
    def on_epoch_end(self, epoch, logs=None):
        """Save model every N epochs"""
        if (epoch + 1) % self.save_every_n_epochs == 0:
            # Format filename with epoch number
            save_path = self.filepath.format(epoch=epoch+1)
            self.model.save(save_path)
            self.logger.info(f"Saved checkpoint at epoch {epoch+1}: {save_path}")
