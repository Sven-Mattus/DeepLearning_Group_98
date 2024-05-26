
import tensorflow as tf
from keras.src import layers

from neural_network.transformer.CustomLayers import Block


class CustomTransformerModel(tf.keras.Model):
    def __init__(self, seq_length, vocab_size, embedding_dim, drop_out_rate, nr_layers, nr_heads):
        super(CustomTransformerModel, self).__init__()
        self.seq_length = seq_length
        self.lay_input_embed = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.lay_pos_embed = layers.Embedding(input_dim=seq_length, output_dim=embedding_dim)
        self.lay_dropout = layers.Dropout(rate=drop_out_rate)
        self.lays_transf_block = [Block(dropout_rate=drop_out_rate, nr_heads=nr_heads, embedding_dim=embedding_dim) for _ in range(nr_layers)]
        self.lay_norm = layers.LayerNormalization()
        self.lm_head = layers.Dense(units=vocab_size, use_bias=False)  # initializer is glorot uniform; units is output_dim

    def build(self, input_shape):
        # Ensure to call the build method of the super class
        super(CustomTransformerModel, self).build(input_shape)

    def call(self, input_idx):  # input_ind of size BATCH_SIZE x SEQ_LENGTH
        pos = tf.range(self.seq_length)[tf.newaxis, :]
        input_emb = self.lay_input_embed(input_idx)  # shape (batch_size, seq_length, embedding_dim)
        pos_emb = self.lay_pos_embed(pos)  # (seq_length, embedding_dim)
        x = self.lay_dropout(input_emb + pos_emb)
        for block in self.lays_transf_block:
            x = block(x)
        x = self.lay_norm(x)
        x = self.lm_head(x)
        return x
