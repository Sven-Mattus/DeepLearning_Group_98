#Idea of the nucleous sampling - source: https://openreview.net/pdf?id=rygGQyrFvH
#In practice this means selecting the highest probability tokens whose cumulative probability mass
#exceeds the pre-chosen threshold p. The size of the sampling set will adjust dynamically based on
#the shape of the probability distribution at each time step. For high values of p, this is a small subset
#of vocabulary that takes up vast majority of the probability mass — the nucleus.


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