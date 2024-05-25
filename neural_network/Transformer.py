import tensorflow as tf

from neural_network.transformer.CustomTransformerModel import CustomTransformerModel


class Transformer:
    def __init__(self, vocab_size, num_layers, embedding_dim, num_heads, batch_size, learning_rate, dropout_rate, seq_length):
        self._model = self._init_model(vocab_size=vocab_size, num_layers=num_layers, embedding_dim=embedding_dim,
                                       num_heads=num_heads, drop_out_rate=dropout_rate, seq_length=seq_length)
        self._model.build(input_shape=(batch_size, None))
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss=self._loss,
            metrics=['accuracy']
        )

    def _loss(self, labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(
            y_true=labels,
            y_pred=logits,
            from_logits=True,
        )

    def _init_model(self, vocab_size, num_layers, embedding_dim, num_heads, drop_out_rate, seq_length):
        return CustomTransformerModel(seq_length=seq_length, vocab_size=vocab_size, embedding_dim=embedding_dim,
                                      drop_out_rate=drop_out_rate, nr_layers=num_layers, nr_heads=num_heads)

    def train_network_with_tf_dataset(self, dataset, nr_epochs, dataset_val):
        history = self._model.fit(
            x=dataset,
            epochs=nr_epochs,
            validation_data=dataset_val,
        )
        return history
