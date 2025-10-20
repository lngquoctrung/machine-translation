import time
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