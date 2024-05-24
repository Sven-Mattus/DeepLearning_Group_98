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
            #THIS PART IS OUR STANDARD SAMPLING STRATEGY
            # predictions = self._model(input_indices)
            # # remove the batch dimension
            # predictions = tf.squeeze(predictions, 1)
            # # Using a categorical distribution to predict the character returned by the model.
            # predictions = predictions / temperature
            # predicted_id = tf.random.categorical(
            #     predictions,
            #     num_samples=1
            # )[-1, 0].numpy()

            #IMPLEMENT NUCLEUOS SAMPLING HERE
            #first, cap the sum of probabilites of the predictions
            p_threshold = 0.5
            batch_elements_indices = []
            batch_elements_probabilites = []

            predictions = self._model(input_indices)
            predictions = tf.squeeze(predictions, 1)
            #the predictions are our raw logits, hence we need to convert them to probabilites
            predictions_probabilities = tf.nn.softmax(predictions)
            prob_array = np.array(predictions_probabilities)
            #prob_sum = np.sum(prob_array)

            #iterate over the batch elements and get the minimum number of probabilites
            for batch_element_i in range(predictions_probabilities.shape[0]):
                batch_element = np.array(predictions_probabilities[batch_element_i])
                probs_batch_element = np.array(prob_array[batch_element_i])
                #returns indices that would sort the array in descending order
                sorted_indices = np.flip(np.argsort(batch_element))

                #get the indices that would just exceed the probabilities
                p_set = 0
                i = 0
                indices_list = []
                while p_set <= p_threshold:
                    index = sorted_indices[i]
                    p_set = p_set + batch_element[index]
                    indices_list.append(index)
                    i+=1

                #check unlinkeli case that p_set = p_threshold
                if p_set == p_threshold:
                    index = sorted_indices[i]
                    p_set = p_set + batch_element[index]
                    indices_list.append(index)

                batch_elements_indices.append(indices_list)
                batch_elements_probabilites = [probs_batch_element[i] for i in batch_elements_indices]
            V_min_set_indices = np.array(batch_elements_indices, dtype='object')
            V_min_set_probabilites = np.array(batch_elements_probabilites, dtype='object')

            # #make independent copies of the tensor and array
            # nucleus_batch_predictions = tf.Tensor()
            # nucleus_predictions = tf.Tensor()
            # for batch_element_i in range(predictions_probabilities.shape[0]):
            #     batch_predictions = predictions[batch_element_i]
            #     for i in V_min_set_indices[batch_element_i]:
            #         nucleus_batch_predictions[i]=batch_predictions[i]
            #     nucleus_predictions[batch_element_i] = nucleus_batch_predictions

            # Initialize a TensorArray to collect nucleus_predictions
            nucleus_predictions = tf.TensorArray(dtype=tf.float32, size=predictions_probabilities.shape[0])

            # Iterate over batch elements
            for batch_element_i in range(predictions_probabilities.shape[0]):
                batch_predictions = predictions[batch_element_i]
                nucleus_batch_predictions = tf.zeros_like(batch_predictions)
                
                # Update nucleus_batch_predictions based on V_min_set_indices
                for i in V_min_set_indices[batch_element_i]:
                    nucleus_batch_predictions = tf.tensor_scatter_nd_update(
                        nucleus_batch_predictions,
                        indices=[[i]],
                        updates=[batch_predictions[i]]
                    )
                
                # Write the updated predictions to the TensorArray
                nucleus_predictions = nucleus_predictions.write(batch_element_i, nucleus_batch_predictions)

            # Stack the TensorArray to get the final nucleus_predictions tensor
            nucleus_predictions = nucleus_predictions.stack()

            predicted_id = tf.random.categorical(
                nucleus_predictions,
                num_samples=1
            )[-1, 0].numpy()
            #sort all character predictions by likelyhood value

            #sum all up until p_threshold is exceeded

            #second, only sample among the reduced character set

            # We pass the predicted character as the next input to the model
            # along with the previous hidden state.
            input_indices = tf.expand_dims([predicted_id], 0)
            charr = data_converter.ind_to_char(predicted_id)
            text_generated += str(charr)

        return start_string + text_generated
    
    def save_weights(self, filename):
        filepath = f'results/weights/{filename}.weights.h5'
        self._model.save_weights(filepath)
