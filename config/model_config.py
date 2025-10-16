from pathlib import Path
root_dir = Path(__file__).parent.parent.absolute()

class ModelConfig:
    """Configuration for BiLSTM Attention model"""

    # Vocabulary
    MAX_VOCAB_SIZE_SRC = 57339
    MAX_VOCAB_SIZE_TRG = 40934

    # Model architecture
    EMBEDDING_DIM = 128
    LSTM_UNITS = 256
    ATTENTION_HEADS = 1

    # Training
    BATCH_SIZE = 128
    EPOCHS = 150
    LEARNING_RATE = 0.001

    # Sequence lengths
    MAX_LENGTH_SRC = 49
    MAX_LENGTH_TRG = 63

    # Paths
    DATA_PATH = str(root_dir / 'data')
    MODEL_SAVE_PATH = str(root_dir / "models" / "saved_models")
    CHECKPOINT_PATH = str(root_dir / "models" / "checkpoints")
    TOKENIZER_PATH = str(root_dir / "models" / "tokenizers")

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {
            "max_vocab_size_src": cls.MAX_VOCAB_SIZE_SRC,
            "max_vocab_size_trg": cls.MAX_VOCAB_SIZE_TRG,
            "embedding_dim": cls.EMBEDDING_DIM,
            "lstm_units": cls.LSTM_UNITS,
            "batch_size": cls.BATCH_SIZE,
            "epochs": cls.EPOCHS,
        }