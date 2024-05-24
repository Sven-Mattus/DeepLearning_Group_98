import numpy as np
import tensorflow as tf


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, _ = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)

        return output, attention_weights


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_size, num_heads, dff, drop_out_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=embedding_size)
        # self.mha = MultiHeadSelfAttention(embedding_size, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(embedding_size)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.add1 = tf.keras.layers.Add()
        self.add2 = tf.keras.layers.Add()

    def call(self, x, training):

        attn_output = self.mha(query=x, value=x, key=x, training=training, use_causal_mask=True)
        x = self.add1([x, attn_output])
        x = self.layernorm1(x)
        mlp_output = self.ffn(x)
        x = self.add2([x, mlp_output])
        x = self.layernorm2(x)
        return x


class PositionEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, embedding_dim):
        super(PositionEmbeddingLayer, self).__init__()
        self.pos_encoding = self.positional_encoding(max_seq_len, embedding_dim)

    def positional_encoding(self, position, embedding_size):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(embedding_size)[np.newaxis, :],
                                     embedding_size)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]  # pos encoding wird zeilenweise drauf addiert

class Transformer:
    def __init__(self, vocab_size, num_layers, embedding_dim, num_heads, dff, batch_size, learning_rate=0.01, rate=0.1):
        self._model = self._init_model(vocab_size, num_layers, embedding_dim, num_heads, dff, rate)
        self._model.build(input_shape=(batch_size, None))
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self._loss,
            metrics=['accuracy']
        )

    def _loss(self, labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(
            y_true=labels,
            y_pred=logits,
            from_logits=True,
        )

    def _init_model(self, vocab_size, num_layers, embedding_dim, num_heads, dff, drop_out_rate):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
        ))
        model.add(PositionEmbeddingLayer(vocab_size, embedding_dim))
        for _ in range(num_layers):
            model.add(TransformerBlock(embedding_dim, num_heads, dff, drop_out_rate))
        model.add(tf.keras.layers.Dropout(drop_out_rate))
        model.add(tf.keras.layers.Dense(vocab_size))
        return model

    def train_network_with_tf_dataset(self, dataset, nr_epochs, dataset_val):
        history = self._model.fit(
            x=dataset,
            epochs=nr_epochs,
            validation_data=dataset_val,
        )
        return history
