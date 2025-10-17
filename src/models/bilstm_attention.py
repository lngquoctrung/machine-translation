import tensorflow as tf

from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM,
    Dense, MultiHeadAttention, concatenate
)
from tensorflow.keras.models import Model

class BiLSTMAttentionModel:
    """BiLSTM with Multi-Head Attention for Machine Translation"""
    def __int__(self, config):
        self.config = config

    def build(self, vocab_size_src, vocab_size_trg, max_len_src, max_len_trg):
        """Build the complete model"""
        # Encoder
        encoder_input = Input(shape=(max_len_src,), name='encoder_input')
        encoder_embedding = Embedding(
            vocab_size_src,
            self.config['embedding_dim'],
            mask_zero=True,
            name='encoder_embedding'
        )(encoder_input)

        # BiLSTM
        encoder_bilstm = Bidirectional(
            LSTM(
                self.config['lstm_units'],
                dropout=0.2,
                return_sequences=True,
                return_state=True,
                name='encoder_lstm'
            ),
            name='encoder_bilstm'
        )(encoder_embedding)

        encoder_output = encoder_bilstm[0]
        # Forward states
        fwd_h, fwd_c = encoder_bilstm[1], encoder_bilstm[2]
        # Backward states
        bwd_h, bwd_c = encoder_bilstm[3], encoder_bilstm[4]

        # Concatenate states
        h = concatenate([fwd_h, bwd_h])
        c = concatenate([fwd_c, bwd_c])
        encoder_states = [h, c]

        # Decoder
        decoder_input = Input(shape=(max_len_trg,), name='decoder_input')
        decoder_embedding = Embedding(
            vocab_size_trg,
            self.config['embedding_dim'],
            mask_zero=True,
            name='decoder_embedding'
        )(decoder_input)

        # LSTM decoder
        decoder_lstm = LSTM(
            self.config['lstm_units'] * 2,
            dropout=0.2,
            return_sequences=True,
            return_state=True,
            name='decoder_lstm'
        )
        decoder_outputs, _, _ = decoder_lstm(
            decoder_embedding,
            initial_state=encoder_states
        )

        # Attention
        attention_layer = MultiHeadAttention(
            key_dim=self.config['lstm_dim'],
            num_heads=1,
            name='attention_layer'
        )
        attention_output, attention_scores = attention_layer(
            query=decoder_outputs,
            value=encoder_output,
            return_attention_scores=True
        )

        # Concatenate attention output with decoder output
        concat_output = concatenate([decoder_outputs, attention_output])

        # Dense layer
        decoder_dense = Dense(
            vocab_size_trg,
            activation='softmax',
            name='decoder_dense'
        )
        decoder_outputs = decoder_dense(concat_output)

        # Build complete model
        model = Model(
            inputs=[encoder_input, decoder_input],
            outputs=decoder_outputs,
            name='bilstm_attention_model'
        )

        return model
