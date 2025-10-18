import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, LSTM,
    Dense, MultiHeadAttention, concatenate,
    Dropout, LayerNormalization, Add
)
from tensorflow.keras.models import Model
from src.utils.logger import setup_logger

class LSTMAttentionModel:
    """
    LSTM (Uni-directional) + Multi-Head Attention
    """

    def __init__(self, config, name_logger=__name__, filename_logger=None):
        self.config = config
        self.logger = setup_logger(
            name=name_logger,
            log_file=filename_logger
        )
        self.model = None

    def build(self, vocab_size_src, vocab_size_trg, max_len_src, max_len_trg):
        """Build LSTM Attention model"""
        # ========== ENCODER ==========
        encoder_inputs = Input(shape=(max_len_src,), name="encoder_input")

        # Embedding
        encoder_embedding = Embedding(
            vocab_size_src,
            self.config["embedding_dim"],
            mask_zero=True,
            embeddings_initializer="glorot_uniform",
            name="encoder_embedding"
        )(encoder_inputs)

        # Layer Normalization
        encoder_embedding = LayerNormalization(name="encoder_ln1")(encoder_embedding)
        encoder_embedding = Dropout(0.2)(encoder_embedding)

        # LSTM Encoder
        encoder_lstm = LSTM(
            self.config["lstm_units"],
            dropout=0.2,
            recurrent_dropout=0.1,
            return_sequences=True,
            return_state=True,
            kernel_initializer="glorot_uniform",
            name="encoder_lstm"
        )
        encoder_outputs, h, c = encoder_lstm(encoder_embedding)

        encoder_outputs = LayerNormalization(name="encoder_ln2")(encoder_outputs)
        encoder_states = [h, c]

        # ========== DECODER ==========
        decoder_inputs = Input(shape=(max_len_trg,), name="decoder_input")

        # Embedding
        decoder_embedding = Embedding(
            vocab_size_trg,
            self.config["embedding_dim"],
            mask_zero=True,
            embeddings_initializer="glorot_uniform",
            name="decoder_embedding"
        )(decoder_inputs)

        decoder_embedding = LayerNormalization(name="decoder_ln1")(decoder_embedding)
        decoder_embedding = Dropout(0.2)(decoder_embedding)

        # LSTM Decoder
        decoder_lstm = LSTM(
            self.config["lstm_units"],
            dropout=0.2,
            recurrent_dropout=0.1,
            return_sequences=True,
            return_state=True,
            kernel_initializer="glorot_uniform",
            name="decoder_lstm"
        )
        decoder_outputs, _, _ = decoder_lstm(
            decoder_embedding,
            initial_state=encoder_states
        )

        decoder_outputs = LayerNormalization(name="decoder_ln2")(decoder_outputs)

        # ========== MULTI-HEAD ATTENTION ==========
        attention_layer = MultiHeadAttention(
            key_dim=self.config['lstm_units'],
            num_heads=self.config.get("attention_heads", 2),
            dropout=0.1,
            name="attention_layer"
        )

        attention_output = attention_layer(
            query=decoder_outputs,
            value=encoder_outputs,
            key=encoder_outputs
        )

        attention_output = LayerNormalization(name="attention_ln")(attention_output)

        # ========== RESIDUAL CONNECTION ==========
        decoder_outputs_residual = Add(name="residual_add")([
            decoder_outputs,
            attention_output
        ])
        decoder_outputs_residual = LayerNormalization(name="residual_ln")(
            decoder_outputs_residual
        )

        # Concatenate
        concat_output = concatenate([
            decoder_outputs_residual,
            attention_output
        ], name="concat_output")

        concat_output = Dropout(0.3)(concat_output)

        # ========== OUTPUT LAYER ==========
        outputs = Dense(
            vocab_size_trg,
            activation="softmax",
            dtype="float32",
            kernel_initializer="glorot_uniform",
            name="decoder_dense"
        )(concat_output)

        # Build model
        self.model = Model(
            [encoder_inputs, decoder_inputs],
            outputs,
            name="lstm_attention"
        )

        return self.model

    def summary(self):
        """Print summary of the model"""
        if self.model is None:
            raise ValueError("Model must be built")
        self.model.summary()

    def count_params(self):
        """Count the number of parameters in the model"""
        if self.model is None:
            raise ValueError("Model must be built")
        return self.model.count_params()

    def save(self, filepath):
        """Save the model"""
        if self.model is None:
            raise ValueError("Model must be built")
        self.model.save(filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load the model"""
        self.model = tf.keras.models.load_model(filepath)
        return self.model
