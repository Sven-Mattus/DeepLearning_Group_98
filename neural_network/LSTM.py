import tensorflow as tf

from data_handler.DataConverter import DataConverter


class LSTM:

    def __init__(self, vocab_size, embedding_dim, nr_rnn_units, batch_size):
        self._model = self._init_model(vocab_size, embedding_dim, nr_rnn_units)
        self._model.build(input_shape=(batch_size, None))
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=self._loss,
            metrics = ['accuracy']
        )

    def train_network(self, dataset_input, dataset_target, nr_epochs, batch_size, val_input, val_target):
        history = self._model.fit(
            x=dataset_input,
            y=dataset_target,
            epochs=nr_epochs,
            batch_size=batch_size,
            # We pass some validation for monitoring validation loss and metrics at the end of each epoch
            validation_data=(val_input, val_target),
            # callbacks=[callback]
        )
        return history

    def train_network_with_tf_dataset(self, dataset, nr_epochs, dataset_val):
        history = self._model.fit(
            x=dataset,
            epochs=nr_epochs,
            validation_data=dataset_val,
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
            from_logits=True,
        )
    
    def evaluate(self, x, y, bs):
        tl, acc = self._model.evaluate(x, y, bs)
        return tl, acc

    def generate_text(self, temperature, start_string, data_converter: DataConverter, num_generate=1000):
        input_indices = data_converter.chars_to_ind(start_string)
        input_indices = tf.expand_dims(input_indices, 0)
        text_generated = ""
        # Here batch size == 1.
        for char_index in range(num_generate):
            predictions = self._model(input_indices)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 1)
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
