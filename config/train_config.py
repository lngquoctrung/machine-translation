from pathlib import Path
root_dir = Path(__file__).parent.parent.absolute()

class TrainConfig:
    """Training hyperparameters"""
    BATCH_SIZE = 128
    EPOCHS = 100
    VALIDATION_SPLIT = 0.1

    # Optimizer
    LEARNING_RATE = 1e-3
    BETA_1 = 0.9
    BETA_2 = 0.98
    EPSILON = 1e-9

    # Learning rate schedule
    USE_LR_SCHEDULER = True
    WARMUP_STEPS = 4000

    # Early stopping
    EARLY_STOPPING_PATIENCE = 5
    REDUCE_LR_PATIENCE = 3
    REDUCE_LR_FACTOR = 0.5
    MIN_LR = 1e-6

    # Callbacks
    SAVE_BEST_ONLY = True
    MONITOR = "val_loss"

    # Logging
    LOG_DIR = str(root_dir / "logs")
    TENSORBOARD_UPDATE_PREQ = "epoch"