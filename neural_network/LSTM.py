import tensorflow as tf
import numpy as np

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

    def _loss(self, labels, logits, reduction='sum'):
        #scce = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
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
            )[-1, 0].np()

            # We pass the predicted character as the next input to the model
            # along with the previous hidden state.
            input_indices = tf.expand_dims([predicted_id], 0)
            charr = data_converter.ind_to_char(predicted_id)
            text_generated += str(charr)

        return start_string + text_generated
    

    def generate_text_nucleus(self, temperature, start_string, data_converter: DataConverter, threshold=0.9, num_generate=1000):
        input_indices = data_converter.chars_to_ind(start_string)
        input_indices = tf.expand_dims(input_indices, 0)
        text_generated = ""
        for _ in range(num_generate):
            predictions = self._model(input_indices)
            predictions = tf.squeeze(predictions, 1) / temperature

            sorted_predictions = np.argsort(predictions[0])[::-1]
            cumulative_probs = np.cumsum(np.sort(predictions[0])[::-1])
            cutoff_index = np.searchsorted(cumulative_probs, threshold)
            truncated_predictions = sorted_predictions[:cutoff_index + 1]
            probabilities = predictions[0][truncated_predictions[0]]
            probabilities = probabilities / np.sum(probabilities)
            predicted_id = np.random.choice(truncated_predictions, p=probabilities)
            input_indices = tf.expand_dims([predicted_id], 0)
            charr = data_converter.ind_to_char(predicted_id)
            text_generated += str(charr)
        return start_string + text_generated

    # def generate_text_nucleus_sampling(self, temperature, start_string, data_converter: DataConverter, num_generate=1000, p=0.9):
    #     input_indices = data_converter.chars_to_ind(start_string)
    #     input_indices = tf.expand_dims(input_indices, 0)
    #     text_generated = ""

    #     for char_index in range(num_generate):
    #         predictions = self._model(input_indices)
    #         predictions = tf.squeeze(predictions, 1)
    #         predictions = predictions / temperature

    #         # Apply nucleus sampling
    #         sorted_indices = tf.argsort(predictions, direction='DESCENDING')
    #         sorted_predictions = tf.sort(predictions, direction='DESCENDING')
    #         cumulative_probs = tf.math.cumsum(tf.nn.softmax(sorted_predictions))

    #         # Exclude tokens with cumulative probability above the threshold p
    #         sorted_indices_to_keep = sorted_indices[cumulative_probs <= p]
    #         chosen_index = tf.random.categorical(tf.expand_dims(tf.gather(predictions, sorted_indices_to_keep), 0), num_samples=1)[0, 0]

    #         predicted_id = sorted_indices_to_keep[chosen_index].np()

    #         input_indices = tf.expand_dims([predicted_id], 0)
    #         text_generated += data_converter.ind_to_char(predicted_id)

    #     return start_string + text_generated

    # def nucleus_sampling(logits, temperature=1.0, threshold=0.95):
    #     # Apply temperature scaling
    #     scaled_logits = logits / temperature
    #     probabilities = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))

    #     # Sort probabilities to identify the cutoff for the nucleus
    #     sorted_indices = np.argsort(probabilities)[::-1]
    #     sorted_probs = probabilities[sorted_indices]
    #     cumulative_probs = np.cumsum(sorted_probs)
        
    #     # Determine the cutoff index for the nucleus such that cumulative probability is within the threshold
    #     cutoff_index = np.where(cumulative_probs > threshold)[0][0]
        
    #     # Consider probabilities only up to the cutoff index, re-normalize to form a valid probability distribution
    #     effective_probs = np.zeros_like(probabilities)
    #     effective_probs[sorted_indices[:cutoff_index + 1]] = probabilities[sorted_indices[:cutoff_index + 1]]
    #     effective_probs /= np.sum(effective_probs)
        
    #     # Sample from the effective probability distribution
    #     sampled_index = np.random.choice(len(effective_probs), p=effective_probs)
    #     return sampled_index


    # def generate_text_nucleus(self, temperature, start_string, data_converter, threshold=0.95):
    #     generated_text = start_string
    #     input_seq = data_converter.chars_to_ind([c for c in start_string])
    #     input_seq = np.array(input_seq).reshape(1, -1)
        
    #     for _ in range(400):  # or whatever length of text you want to generate
    #         predictions = self._model.predict(input_seq)
    #         next_char_index = self.nucleus_sampling(predictions[0], temperature, threshold)
    #         next_char = data_converter.ind_to_chars(next_char_index)
    #         generated_text += next_char
    #         input_seq = np.append(input_seq[0], next_char_index)[1:].reshape(1, -1)

    #     return generated_text
