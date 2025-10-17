from pathlib import Path
root_dir = Path(__file__).parent.parent.absolute()

class ModelConfig:
    """Configuration for BiLSTM Attention model"""

    # Vocabulary size
    MAX_VOCAB_SIZE_SRC = 25000      # Limit to save memory
    MAX_VOCAB_SIZE_TRG = 20000      # Limit to save memory
    MIN_WORD_PREQUENCY = 2          # Filter rare words

    # Model architecture
    EMBEDDING_DIM = 64              # To save memory
    LSTM_UNITS = 128                # Hidden units
    ATTENTION_HEADS = 2             # Using multi-head for accuracy

    # Training
    BATCH_SIZE = 256                # Increase training speed
    EPOCHS = 100
    LEARNING_RATE = 0.001

    # Sequence lengths
    MAX_LENGTH_SRC = 40             # Limit the number of words in a source sequence
    MAX_LENGTH_TRG = 50             # Limit the number of word in a target sequence

    # GPU settings
    USE_MIXED_PRECISION = True      # Save memory, increase training speed
    GPU_MEMORY_GROWTH = True
    GPU_MEMORY_LIMIT = 15000        # Limit 15GB GPU to avoid out off memory

    # Accuracy improvements
    LABEL_SMOOTHING = 0.1           # Decrease overconfidence
    USE_LAYER_NORM = True           # Keep traning stably
    USE_LR_SCHEDULER = True         # Warmup + cosin decay
    WARMUP_STEPS = 4000
    MIN_LR = 1e-6

    # Inference
    BEAM_WIDTH = 5                  # Beam search for better accuracy
    USE_BEM_SEARCH = True

    # Callback
    EARLY_STOPPING_PATIENCE = 5
    REDUCE_LR_PATIENCE = 3

    # Paths
    DATA_PATH = str(root_dir / "data")
    MODEL_SAVE_PATH = str(root_dir / "models" / "saved_models")
    CHECKPOINT_PATH = str(root_dir / "models" / "checkpoints")
    TOKENIZER_PATH = str(root_dir / "models" / "tokenizers")
    LOG_DIR = str(root_dir / "logs")

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {
            "max_vocab_size_src": cls.MAX_VOCAB_SIZE_SRC,
            "max_vocab_size_trg": cls.MAX_VOCAB_SIZE_TRG,
            "embedding_dim": cls.EMBEDDING_DIM,
            "lstm_units": cls.LSTM_UNITS,
            "attention_heads": cls.ATTENTION_HEADS,
            "batch_size": cls.BATCH_SIZE,
            "epochs": cls.EPOCHS,
            "max_length_src": cls.MAX_LENGTH_SRC,
            "max_length_trg": cls.MAX_LENGTH_TRG,
            "use_mixed_precision": cls.USE_MIXED_PRECISION,
            "use_lr_scheduler": cls.USE_LR_SCHEDULER,
            "learning_rate": cls.LEARNING_RATE,
            "warmup_steps": cls.WARMUP_STEPS,
            "label_smoothing": cls.LABEL_SMOOTHING,
            "beam_width": cls.BEAM_WIDTH,
            "use_beam_search": cls.USE_BEM_SEARCH,
            "early_stopping_patience": cls.EARLY_STOPPING_PATIENCE,
            "checkpoint_path": cls.CHECKPOINT_PATH,
        }

    @classmethod
    def estimate_memory(cls):
        """Estimate the memory usage of the model"""
        embedding_params_src = cls.MAX_LENGTH_SRC * cls.EMBEDDING_DIM
        embedding_params_trg = cls.MAX_LENGTH_TRG * cls.EMBEDDING_DIM

        # Encoder block => Formula: 4 * H * (E * H) * 2
        # Decoder block => Formula: 4 * H * 2 * (E + H * 2)
        # Where:
        #   H = LSTM hidden size
        #   E = input size (embedding dim)
        #   4 = 4 gates (input, forget, output and cell)
        #   2 = bidirectional
        lstm_params = (
                4 * cls.LSTM_UNITS * (cls.EMBEDDING_DIM + cls.LSTM_UNITS) * 2 +
                4 * cls.LSTM_UNITS * 2 * (cls.EMBEDDING_DIM + cls.LSTM_UNITS * 2)
        )
        dense_params = cls.LSTM_UNITS * 2 * cls.MAX_VOCAB_SIZE_TRG
        total_params = embedding_params_src + embedding_params_trg + lstm_params + dense_params

        bytes_per_params = 2 if cls.USE_MIXED_PRECISION else 4
        memory_mb = (total_params * bytes_per_params * 3) / (1024 * 1024)

        batch_memory = (
            cls.BATCH_SIZE *
            (cls.MAX_LENGTH_SRC * cls.MAX_VOCAB_SIZE_TRG) *
            cls.EMBEDDING_DIM *
            bytes_per_params / (1024 * 1024)
        )
        total_memory = memory_mb + batch_memory

        return {
            'total_params': f'{total_params:,}',
            'model_memory_mb': f'{memory_mb:.2f} MB',
            'batch_memory_mb': f'{batch_memory:.2f} MB',
            'total_memory_gb': f'{total_memory / 1024:.2f} GB'
        }
