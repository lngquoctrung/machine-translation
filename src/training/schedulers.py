import tensorflow as tf
import numpy as np

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning Rate Schedule: Linear Warmup + Cosine Decay
        - During warmup: LR increases linearly from 0 → peak_lr
        - After warmup: Cosine decay from peak_lr → min_lr

    Parameters:
        peak_lr (float): highest learning rate after warmup
        warmup_steps (int): number of steps to warmup
        total_steps (int): total number of training steps
        min_lr (float): minimum learning rate after decay
    """
    def __init__(self, peak_lr, warmup_steps, total_steps, min_lr=1e-6):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr


    def __call__(self, step):
        """Compute LR for given step"""
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        # Warmup phase
        warmup_lr = (step / warmup_steps) * self.peak_lr

        # Cosine decay phase
        decay_steps = total_steps - warmup_steps
        decay_step = step - warmup_steps

        cosine_decay = 0.5 * (1 + tf.cos(np.pi * decay_step / decay_steps))
        decay_lr = (self.peak_lr - self.min_lr) * cosine_decay + self.min_lr

        # Return appropriate LR
        lr = tf.cond(
            step < warmup_steps,
            lambda: warmup_lr,
            lambda: decay_lr
        )

        return lr

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr
        }

def get_lr_schedule(schedule_type='warmup_cosine', **kwargs):
    """
    Factory function for LR schedule

    Parameters:
        schedule_type (str): 'warmup_cosine' | 'constant'
    """
    if schedule_type == 'warmup_cosine':
        return WarmupCosineDecay(
            peak_lr=kwargs.get('peak_lr', 0.001),
            warmup_steps=kwargs.get('warmup_steps', 4000),
            total_steps=kwargs.get('total_steps', 100000),
            min_lr=kwargs.get('min_lr', 1e-6)
        )
    elif schedule_type == 'constant':
        return kwargs.get('learning_rate', 0.001)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")