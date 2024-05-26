import tensorflow as tf

from data_handler.DataConverter import DataConverter
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

    def train_network(self, dataset_input, dataset_target, nr_epochs, batch_size, val_input, val_target):
        history = self._model.fit(
            x=dataset_input,
            y=dataset_target,
            epochs=nr_epochs,
            batch_size=batch_size,
            # We pass some validation for monitoring validation loss and metrics at the end of each epoch
            validation_data=(val_input, val_target),
        )
        return history

    def train_network_with_tf_dataset(self, dataset, nr_epochs, dataset_val):
        history = self._model.fit(
            x=dataset,
            epochs=nr_epochs,
            validation_data=dataset_val,
        )
        return history


    def generate_text(self, temperature, start_string, data_converter: DataConverter, num_generate=1000):
        input_indices = data_converter.chars_to_ind(start_string)
        input_indices = tf.expand_dims(input_indices, 0)
        text_generated = ""
        # Here batch size == 1.
        for char_index in range(num_generate):
            # THIS PART IS OUR STANDARD SAMPLING STRATEGY
            predictions = self._model(input_indices)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)
            # Using a categorical distribution to predict the character returned by the model.
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(
                predictions,
                num_samples=1
            )[-1, 0].numpy()
            # We pass the predicted character as the next input to the model
            # along with the previous hidden state.
            input_indices = tf.expand_dims([predicted_id], 0)
            charr = data_converter.ind_to_char(predicted_id)
            text_generated += str(charr)
        return start_string + text_generated

