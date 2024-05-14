
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


class LSTM:

    def __init__(self, vocab_size, embedding_dim, nr_rnn_units, batch_size):
        self._model = self._init_model(vocab_size, embedding_dim, nr_rnn_units)
        self._model.build(input_shape=(batch_size, None))
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=self._loss
        )

    def train_network(self, dataset_input, dataset_target, nr_epochs, batch_size):
        history = self._model.fit(
            x=dataset_input,
            y=dataset_target,
            epochs=nr_epochs,
            batch_size=batch_size,
            shuffle=False  # todo remove
        )
        return history

    def train_network_with_tf_dataset(self, dataset, nr_epochs):
        history = self._model.fit(
            x=dataset,
            epochs=nr_epochs,
        )
        return history

    def _init_model(self, vocab_size, embedding_dim, nr_rnn_units):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
        ))
        model.add(tf.keras.layers.LSTM(
            units=nr_rnn_units,
            return_sequences=True,
            stateful=True,
            recurrent_initializer=tf.keras.initializers.GlorotNormal()
        ))
        model.add(tf.keras.layers.Dense(vocab_size))
        return model

    def _loss(self, labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(
            y_true=labels,
            y_pred=logits,
            from_logits=True
        )

