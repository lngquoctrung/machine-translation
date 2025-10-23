from pathlib import Path

root_dir = Path(__file__).parent.parent.absolute()

class Config:
    BATCH_SIZE = 128                      # Increase training speed
    EPOCHS = 100
    VALIDATION_SPLIT = 0.1
    LAYER_DROPOUT = 0.2
    LSTM_DROPOUT = 0.2
    ATTENTION_DROPOUT = 0.1
    LN_EPSILON = 1e-6

    # Optimizer
    LEARNING_RATE = 0.001
    BETA_1 = 0.9
    BETA_2 = 0.98
    OPTIMIZER_EPSILON = 1e-9
    OPTIMIZER_DYNAMIC = True
    OPTIMIZER_INITIAL_SCALE = 2**15
    OPTIMIZER_DYNAMIC_GROWTH_STEPS = 2000

    # Learning rate schedule
    USE_LR_SCHEDULER = True              # Warmup + cosin decay
    WARMUP_STEPS = 8000
    TOTAL_STEPS = 100000
    MIN_LR = 1e-7

    # Early stopping
    EARLY_STOPPING_PATIENCE = 5
    REDUCE_LR_PATIENCE = 3
    REDUCE_LR_FACTOR = 0.5

    # Callbacks
    SAVE_BEST_ONLY = True
    MONITOR = "val_loss"

    # ==========================
    # Configuration for models
    # ==========================
    MAX_LENGTH_SRC = 40                  # Limit the number of words in a source sequence
    MAX_LENGTH_TRG = 50                  # Limit the number of word in a target sequence
    MAX_VOCAB_SIZE_SRC = 30000           # Limit to save memory
    MAX_VOCAB_SIZE_TRG = 25000           # Limit to save memory
    MIN_WORD_FREQUENCY = 1               # Filter rare words

    EMBEDDING_DIM = 64                   # To save memory
    LSTM_UNITS = 128                     # Hidden units
    ATTENTION_HEADS = 4                  # Using multi-head for accuracy

    # GPU settings
    USE_MIXED_PRECISION = True           # Save memory, increase training speed
    GPU_MEMORY_GROWTH = True
    GPU_MEMORY_LIMIT = 15000             # Limit 15GB GPU to avoid out off memory

    # Accuracy improvements
    LABEL_SMOOTHING = 0.05               # Decrease overconfidence
    USE_LAYER_NORM = True                # Keep traning stably

    # Inference
    BEAM_WIDTH = 5                       # Beam search for better accuracy
    USE_BEAM_SEARCH = True

    # Paths
    DATA_PATH = str(root_dir / "data")
    ARTIFACT_PATH = str(root_dir / "artifacts")
    ASSETS_PATH = str(root_dir / "assets")
    LOG_DIR = str(root_dir / "logs")
    TENSORBOARD_UPDATE_FREQ = "epoch"

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {
            # Training
            "batch_size": cls.BATCH_SIZE,
            "epochs": cls.EPOCHS,
            "validation_split": cls.VALIDATION_SPLIT,

            # Optimizer
            "learning_rate": cls.LEARNING_RATE,
            "beta_1": cls.BETA_1,
            "beta_2": cls.BETA_2,
            "optimizer_epsilon": cls.OPTIMIZER_EPSILON,
            "optimizer_dynamic": cls.OPTIMIZER_DYNAMIC,
            "optimizer_initial_scale": cls.OPTIMIZER_INITIAL_SCALE,
            "optimizer_dynamic_growth_steps": cls.OPTIMIZER_DYNAMIC_GROWTH_STEPS,

            # Learning rate schedule
            "use_lr_scheduler": cls.USE_LR_SCHEDULER,
            "warmup_steps": cls.WARMUP_STEPS,
            "total_steps": cls.TOTAL_STEPS,
            "min_lr": cls.MIN_LR,

            # Early stopping
            "early_stopping_patience": cls.EARLY_STOPPING_PATIENCE,
            "reduce_lr_patience": cls.REDUCE_LR_PATIENCE,
            "reduce_lr_factor": cls.REDUCE_LR_FACTOR,

            # Callbacks
            "save_best_only": cls.SAVE_BEST_ONLY,
            "monitor": cls.MONITOR,
            "tensorboard_update_freq": cls.TENSORBOARD_UPDATE_FREQ,

            # Vocabulary
            "max_vocab_size_src": cls.MAX_VOCAB_SIZE_SRC,
            "max_vocab_size_trg": cls.MAX_VOCAB_SIZE_TRG,
            "min_word_frequency": cls.MIN_WORD_FREQUENCY,

            # Model parameters
            "embedding_dim": cls.EMBEDDING_DIM,
            "lstm_units": cls.LSTM_UNITS,
            "attention_heads": cls.ATTENTION_HEADS,
            "layer_dropout": cls.LAYER_DROPOUT,
            "lstm_dropout": cls.LSTM_DROPOUT,
            "attention_dropout": cls.ATTENTION_DROPOUT,
            "ln_epsilon": cls.LN_EPSILON,

            # Sequence limits
            "max_length_src": cls.MAX_LENGTH_SRC,
            "max_length_trg": cls.MAX_LENGTH_TRG,

            # GPU settings
            "use_mixed_precision": cls.USE_MIXED_PRECISION,
            "gpu_memory_growth": cls.GPU_MEMORY_GROWTH,
            "gpu_memory_limit": cls.GPU_MEMORY_LIMIT,

            # Accuracy improvements
            "label_smoothing": cls.LABEL_SMOOTHING,
            "use_layer_norm": cls.USE_LAYER_NORM,

            # Inference
            "beam_width": cls.BEAM_WIDTH,
            "use_beam_search": cls.USE_BEAM_SEARCH,
        }

    @classmethod
    def estimate_memory(cls):
        """Estimate the memory usage of the model"""
        embedding_params_src = cls.MAX_VOCAB_SIZE_SRC * cls.EMBEDDING_DIM
        embedding_params_trg = cls.MAX_VOCAB_SIZE_TRG * cls.EMBEDDING_DIM

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
