import math

from keras.src import layers
import tensorflow as tf


class Block(layers.Layer):

    def __init__(self, dropout_rate, nr_heads, embedding_dim):
        super().__init__()
        self.attn = CausalSelfAttention(dropout_rate, nr_heads, embedding_dim)
        self.lay_norm_1 = layers.LayerNormalization()
        self.mlp = MLP(embedding_dim, dropout_rate)
        self.lay_norm_2 = layers.LayerNormalization()

    def build(self, input_shape):
        super(Block, self).build(input_shape)

    def call(self, x):
        x = x + self.attn(x)
        x = self.lay_norm_1(x)
        x = x + self.mlp(x)
        x = self.lay_norm_2(x)
        return x

class EncodeAndDecodeBlock(layers.Layer):

    def __init__(self, dropout_rate, nr_heads, embedding_dim):
        super().__init__()
        self.attn = CausalSelfAttention(dropout_rate, nr_heads, embedding_dim)
        self.lay_norm_1 = layers.LayerNormalization()
        self.mlp_enc = MLP(embedding_dim, dropout_rate)
        self.mlp_dec = MLP(embedding_dim, dropout_rate)
        self.lay_norm_2 = layers.LayerNormalization()
        self.lay_norm_3 = layers.LayerNormalization()
        self.lay_norm_4 = layers.LayerNormalization()
        self.lay_norm_5 = layers.LayerNormalization()
        self.lay_key = layers.Dense(units=embedding_dim, use_bias=False)
        self.lay_value = layers.Dense(units=embedding_dim, use_bias=False)
        self.lay_query = layers.Dense(units=embedding_dim, use_bias=False)
        key_dim = embedding_dim // nr_heads
        self.lay_multihead_att_enc = layers.MultiHeadAttention(num_heads=nr_heads, key_dim=key_dim, dropout=dropout_rate, use_bias=False)
        self.lay_multihead_att_dec = layers.MultiHeadAttention(num_heads=nr_heads, key_dim=key_dim, dropout=dropout_rate, use_bias=False)
        self.lay_multihead_att_cross = layers.MultiHeadAttention(num_heads=nr_heads, key_dim=key_dim, dropout=dropout_rate, use_bias=False)

    def build(self, input_shape):
        super(EncodeAndDecodeBlock, self).build(input_shape)

    def call(self, input):
        x = input  # [:,0:input.shape[1]-1, :]
        # encode
        q = self.lay_query(x)
        k = self.lay_key(x)
        v = self.lay_value(x)
        att = self.lay_multihead_att_enc(query=q, key=k, value=v, use_causal_mask=True)
        x = x + att
        x = self.lay_norm_1(x)
        x = x + self.mlp_enc(x)
        out_enc = self.lay_norm_2(x)

        # decode
        z = input  # [:,1:,:] todo is not the same as output
        q = self.lay_query(z)
        k = self.lay_key(z)
        v = self.lay_value(z)
        att = self.lay_multihead_att_dec(query=q, key=k, value=v, use_causal_mask=True)
        x = out_enc + att
        x = self.lay_norm_3(x)

        # Cross-attention with encoder output
        q = self.lay_query(x)
        k = self.lay_key(out_enc)
        v = self.lay_value(out_enc)
        att = self.lay_multihead_att_cross(query=q, key=k, value=v, use_causal_mask=False)
        x = x + att
        x = self.lay_norm_4(x)
        x = x + self.mlp_dec(x)
        x = self.lay_norm_5(x)
        return x


class CausalSelfAttention(layers.Layer):

    def __init__(self, dropout_rate, nr_heads, embedding_dim):
        super().__init__()
        self.lay_dropout_1 = layers.Dropout(rate=dropout_rate)
        self.lay_dropout_2 = layers.Dropout(rate=dropout_rate)
        self.nr_heads = nr_heads
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

        self.lay_key = layers.Dense(units=embedding_dim, use_bias=False)
        self.lay_value = layers.Dense(units=embedding_dim, use_bias=False)
        self.lay_query = layers.Dense(units=embedding_dim, use_bias=False)
        # output projection
        self.lay_proj = layers.Dense(units=embedding_dim, use_bias=False)
        key_dim = embedding_dim // nr_heads
        self.lay_multihead_att = layers.MultiHeadAttention(num_heads=nr_heads, key_dim=key_dim, dropout=dropout_rate, use_bias=False)

    def call_old(self, x):
        batch_size, seq_length, embedding_dim = x.shape
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.lay_attn(x)
        q, k, v = tf.split(qkv, num_or_size_splits=3, axis=2)

        # reshape and transpose for multi-head attention
        q = tf.reshape(q, (batch_size, seq_length, self.nr_heads, embedding_dim // self.nr_heads))
        q = tf.transpose(q, perm=[0, 2, 1, 3])  # (batch_size, nr_heads, seq_length, head_dim)

        k = tf.reshape(k, (batch_size, seq_length, self.nr_heads, embedding_dim // self.nr_heads))
        k = tf.transpose(k, perm=[0, 2, 1, 3])  # (batch_size, nr_heads, seq_length, head_dim)

        v = tf.reshape(v, (batch_size, seq_length, self.nr_heads, embedding_dim // self.nr_heads))
        v = tf.transpose(v, perm=[0, 2, 1, 3])  # (batch_size, nr_heads, seq_length, head_dim)

        # y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_rate, is_causal=True)
        att = q @ tf.transpose(k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = tf.matmul(q, k, transpose_b=True) * (1.0 / math.sqrt(embedding_dim // self.nr_heads))
        # mask = tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)  # Lower triangular matrix
        # mask = tf.reshape(mask,
        #                   (1, 1, seq_length, seq_length))  # Shape to (1, 1, seq_length, seq_length) for broadcasting
        #
        # # Apply mask to attention logits
        # att = att * mask + (1.0 - mask) * float('-inf')

        att = tf.nn.softmax(att, axis=-1)
        att = self.lay_dropout_1(att)
        y = tf.matmul(att, v)  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = tf.transpose(y, perm=[0, 2, 1, 3])
        y = tf.reshape(y, (batch_size, seq_length, embedding_dim))  # re-assemble all head outputs side by side

        # output projection
        y = self.lay_proj(y)  # todo is dimension correct
        y = self.lay_dropout_2(y)
        return y

    def call(self, x):
        batch_size, seq_length, embedding_dim = x.shape
        q = self.lay_query(x)
        k = self.lay_key(x)
        v = self.lay_value(x)
        x = self.lay_multihead_att(query=q, key=k, value=v, use_causal_mask=True)

        # single head
        # att = tf.matmul(q, k, transpose_b=True)
        # att = tf.nn.softmax(att, axis=-1)
        # masked_att =
        # x = tf.matmul(att, v)

        return x

    def build(self, input_shape):
        super().build(input_shape)


class MLP(layers.Layer):

    def __init__(self, embedding_dim, dropout_rate):
        super().__init__()
        self.lay_fc = layers.Dense(units=4 * embedding_dim, use_bias=True)
        self.lay_proj = layers.Dense(units=embedding_dim, use_bias=True)
        self.lay_dropout = layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.lay_fc(x)
        x = tf.nn.gelu(x)
        x = self.lay_proj(x)
        x = self.lay_dropout(x)
        return x

    def build(self, input_shape):
        super(MLP, self).build(input_shape)
