import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM,
    Dense, MultiHeadAttention, concatenate,
    Dropout, LayerNormalization, Add
)
from tensorflow.keras.models import Model
from src.utils import setup_logger, sanitize_path

class BiLSTMAttentionModel:
    """BiLSTM with Multi-Head Attention for Machine Translation"""
    def __init__(self, config, name_logger=__name__, filename_logger=None):
        self.config = config
        self.logger = setup_logger(
            name=name_logger,
            log_file=filename_logger
        )
        self.model = None

    def build(self, vocab_size_src, vocab_size_trg, max_len_src, max_len_trg):
        """Build the complete model"""
        # =============== ENCODER ===============
        encoder_input = Input(
            shape=(max_len_src,),
            name='encoder_input',
            dtype = 'int32'
        )

        encoder_embedding = Embedding(
            vocab_size_src,
            self.config["embedding_dim"],
            mask_zero=True,
            embeddings_initializer="glorot_uniform",
            dtype='float32',
            name="encoder_embedding"
        )(encoder_input)

        # Layer Normalization and dropout to avoid overfitting
        encoder_embedding = LayerNormalization(
            epsilon=1e-6,
            name="encoder_ln1"
        )(encoder_embedding)
        encoder_embedding = Dropout(0.2)(encoder_embedding)

        # Encoder BiLSTM
        encoder_bilstm = Bidirectional(
            LSTM(
                self.config["lstm_units"],
                dropout=0.0,
                recurrent_dropout=0.0,
                return_sequences=True,
                return_state=True,
                kernel_initializer="glorot_uniform",
                name="encoder_lstm"
            ),
            name="encoder_bilstm"
        )(encoder_embedding)

        # Normalization
        encoder_outputs = encoder_bilstm[0]
        encoder_outputs = LayerNormalization(
            name="encoder_ln2"
        )(encoder_outputs)

        # Forward states
        fwd_h, fwd_c = encoder_bilstm[1], encoder_bilstm[2]
        # Backward states
        bwd_h, bwd_c = encoder_bilstm[3], encoder_bilstm[4]
        # Concatenate states
        h = concatenate([fwd_h, bwd_h])
        c = concatenate([fwd_c, bwd_c])
        encoder_states = [h, c]

        # =============== DECODER ===============
        decoder_input = Input(
            shape=(max_len_trg,),
            name="decoder_input",
            dtype = 'int32'
        )
        decoder_embedding = Embedding(
            vocab_size_trg,
            self.config["embedding_dim"],
            mask_zero=True,
            embeddings_initializer="glorot_uniform",
            dtype='float32',
            name="decoder_embedding"
        )(decoder_input)

        # Normalization
        decoder_embedding = LayerNormalization(name="decoder_ln1")(decoder_embedding)
        decoder_embedding = Dropout(0.2)(decoder_embedding)

        # LSTM decoder
        decoder_lstm = LSTM(
            self.config["lstm_units"] * 2,
            dropout=0.0,
            recurrent_dropout=0.0,
            return_sequences=True,
            return_state=True,
            kernel_initializer="glorot_uniform",
            name="decoder_lstm"
        )
        decoder_outputs, _, _ = decoder_lstm(
            decoder_embedding,
            initial_state=encoder_states
        )
        # Normalization
        decoder_outputs = LayerNormalization(name="decoder_ln2")(decoder_outputs)

        # Attention Layer
        attention_layer = MultiHeadAttention(
            key_dim=self.config["lstm_units"],
            num_heads=self.config.get("attention_heads", 2),
            dropout=0.1,
            name="attention_layer"
        )
        attention_output = attention_layer(
            query=decoder_outputs,
            value=encoder_outputs,
            key=encoder_outputs
        )

        # Normalization
        attention_output = LayerNormalization(name="attention_ln")(attention_output)

        # Residual connection (add and norm)
        decoder_outputs_residual = Add(name="residual_add")([
            decoder_outputs,
            attention_output
        ])
        decoder_outputs_residual = LayerNormalization(name="residual_ln")(decoder_outputs_residual)

        # Concatenate
        concat_output = concatenate([
            decoder_outputs_residual,
            attention_output
        ], name="concat_output")
        concat_output = Dropout(0.4)(concat_output)

        # Dense layer
        decoder_dense = Dense(
            vocab_size_trg,
            activation="softmax",
            dtype="float32",
            kernel_initializer="glorot_uniform",
            name="decoder_dense"
        )
        decoder_outputs = decoder_dense(concat_output)

        # Build complete model
        self.model = Model(
            inputs=[encoder_input, decoder_input],
            outputs=decoder_outputs,
            name="bilstm_attention_model"
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
        self.logger.info(f"Model saved to {sanitize_path(filepath)}")

    def load(self, filepath):
        """Load the model"""
        self.model = tf.keras.models.load_model(filepath)
        return self.model
