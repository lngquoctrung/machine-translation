import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import Config
from src.utils.gpu_utils import GPUMemoryManager
from src.data.preprocessing import DataPreprocessor
from src.models.bilstm_attention import BiLSTMAttentionModel
from src.models.lstm_attention import LSTMAttentionModel
from src.training.trainer import ModelTrainer
from src.utils.helpers import save_tokenizer, plot_history
import pickle
import time
import argparse


def main(model_type='bilstm'):
    """
    Main training pipeline

    Args:
        model_type: 'lstm' hoặc 'bilstm'
    """
    config = Config.to_dict()
    print(f"MACHINE TRANSLATION TRAINING - {model_type.upper()}")

    # ========== GPU SETUP ==========
    GPUMemoryManager.clear_session()
    GPUMemoryManager.setup_gpu(
        memory_limit_mb=config.get("gpu_memory_limit", 15000),
        allow_growth=config.get("gpu_memory_growth", True)
    )

    if config.get("use_mixed_precision", True):
        GPUMemoryManager.enable_mixed_precision()

    GPUMemoryManager.get_memory_info()

    # ========== CONFIGURATION ==========
    print(f"Config ({model_type.upper()}):")
    for key, value in config.items():
        print(f"   {key}: {value}")

    print("Memory Estimate:")
    for key, value in Config.estimate_memory().items():
        print(f"   {key}: {value}")

    # ========== DATA PREPROCESSING ==========
    preprocessor = DataPreprocessor(
        max_vocab_src=config["max_vocab_size_src"],
        max_vocab_trg=config["max_vocab_size_trg"],
        min_frequency=config.get("min_word_frequency", 2)
    )

    # Load data
    df = preprocessor.load_data(
        src_path=f"{config["data_path"]}/raw/en.txt",
        trg_path=f"{config["data_path"]}/raw/vi.txt",
        max_length_src=config['max_length_src'],
        max_length_trg=config['max_length_trg']
    )
    print(f"Dataset: {df.shape}")

    # Split
    train_df, val_df, test_df = preprocessor.split_data(df)
    print(f"Split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Tokenizers
    tokenizer_src, tokenizer_trg = preprocessor.build_tokenizers(train_df)

    # Save tokenizers
    os.makedirs(config["tokenizer_path"], exist_ok=True)
    save_tokenizer(tokenizer_src, f"{config["tokenizer_path"]}/tokenizer_src.pkl")
    save_tokenizer(tokenizer_trg, f"{config["tokenizer_path"]}/tokenizer_trg.pkl")

    # Sequences
    src_train, trg_in_train, trg_out_train = preprocessor.prepare_sequences(
        train_df, config["max_length_src"], config["max_length_trg"]
    )
    en_val, vi_in_val, vi_out_val = preprocessor.prepare_sequences(
        val_df, config["max_length_src"], config["max_length_trg"]
    )

    print(f"Sequences: {src_train.shape}")

    # ========== BUILD MODEL ==========
    # Choose model type
    if model_type.lower() == "bilstm":
        model_builder = BiLSTMAttentionModel(config)
        print("Building BiLSTM + Attention (Bidirectional)")
    else:
        model_builder = LSTMAttentionModel(config)
        print("Building LSTM + Attention (Uni-directional)")

    model = model_builder.build(
        vocab_size_src=config["max_vocab_size_src"],
        vocab_size_trg=config["max_vocab_size_trg"],
        max_len_src=config["max_length_src"],
        max_len_trg=config["max_length_trg"]
    )

    print(f"{model_type.upper()} Model Summary:")
    model.summary()

    # Compare parameters
    total_params = model.count_params()
    print(f"Total Parameters: {total_params:,}")

    # ========== TRAIN ==========
    steps_per_epoch = len(src_train) // config["batch_size"]
    config["total_steps"] = steps_per_epoch * config["epochs"]

    trainer = ModelTrainer(model=model, config=config)

    start_time = time.time()
    history = trainer.train(
        train_data=(src_train, trg_in_train, trg_out_train),
        val_data=(en_val, vi_in_val, vi_out_val)
    )
    elapsed_time = time.time() - start_time

    # ========== SAVE ==========
    os.makedirs(config["model_save_path"], exist_ok=True)
    model_path = f'{config["model_save_path"]}/{model_type}_model.h5'
    trainer.save_model(model_path)

    with open(f'{config["model_save_path"]}/{model_type}_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    try:
        plot_history(history, save_path=f'{config["assets_path"]}/{model_type}_history.png')
    except:
        print("Could not plot history")

    # ========== SUMMARY ==========
    print(f"{model_type.upper()} TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Time: {elapsed_time / 3600:.2f} hours")
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print(f"Final val_loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Total params: {total_params:,}")
    print(f"Model: {model_path}")

    GPUMemoryManager.get_memory_info()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train translation model')
    parser.add_argument(
        '--model',
        type=str,
        choices=['lstm', 'bilstm'],
        default='bilstm',
        help='Model type: lstm hoặc bilstm (default: bilstm)'
    )

    args = parser.parse_args()
    main(model_type=args.model)
