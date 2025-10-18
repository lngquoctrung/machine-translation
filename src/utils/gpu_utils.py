import os
import tensorflow as tf

class GPUMemoryManager:
    """Manage GPU memory"""
    @staticmethod
    def setup_gpu(memory_limit_mb=None, allow_growth=True):
        """
        Set up GPU with memory optimization

        Args:
            memory_limit_mb: Limit memory (MB)
            allow_growth: Allow memory growth (True) or fixed memory (False)
        """
        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            try:
                for gpu in gpus:
                    if allow_growth:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        print(f"GPU memory growth enabled")

                    if memory_limit_mb:
                        tf.config.set_logical_device_configuration(
                            gpu,
                            [tf.config.LogicalDeviceConfiguration(
                                memory_limit=memory_limit_mb
                            )]
                        )
                        print(f"GPU memory limited to {memory_limit_mb} MB")

                print(f"Found {len(gpus)} GPU(s)")

            except RuntimeError as e:
                print(f"GPU setup error: {e}")
        else:
            print("No GPU found, using CPU")

    @staticmethod
    def enable_mixed_precision():
        """Enable mixed precision training (FP16)"""
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision (FP16) enabled")
        print(f"Compute: {policy.compute_dtype}, Variable: {policy.variable_dtype}")

    @staticmethod
    def clear_session():
        """Clear TensorFlow session"""
        tf.keras.backend.clear_session()
        print("Session cleared")

    @staticmethod
    def get_memory_info():
        """Get GPU memory info"""
        try:
            import subprocess
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                encoding='utf-8'
            )

            for i, line in enumerate(result.strip().split('\n')):
                used, total = line.split(', ')
                print(f"GPU {i}: {used} MB / {total} MB ({float(used) / float(total) * 100:.1f}%)")

        except Exception as e:
            print(f"Cannot get GPU info: {e}")

    @staticmethod
    def set_tf_loglevel(level='ERROR'):
        """Giảm TensorFlow logs"""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = {
            'DEBUG': '0',
            'INFO': '1',
            'WARNING': '2',
            'ERROR': '3'
        }.get(level, '2')
