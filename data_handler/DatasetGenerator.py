import numpy as np
import tensorflow as tf


class DataGenerator:

    @staticmethod
    def create_tf_dataset(book_as_ind, seq_length, batch_size):
        char_dataset = tf.data.Dataset.from_tensor_slices(book_as_ind)
        sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
        dataset = sequences.map(DataGenerator._split_input_target)
        BUFFER_SIZE = len(list(dataset))
        dataset = dataset.shuffle(BUFFER_SIZE).batch(batch_size=batch_size, drop_remainder=True).repeat()
        return dataset

    @staticmethod
    def create_array_dataset(book_as_ind, seq_length):
        nr_seqs_per_epochs = len(book_as_ind) // seq_length  # floor division
        sequences = DataGenerator._chunk_list(book_as_ind, seq_length)
        assert len(sequences) == nr_seqs_per_epochs
        data_block = np.vstack(sequences)
        dataset_input = data_block[:, :seq_length - 1]
        dataset_target = data_block[:, 1:]
        # np.random.shuffle(sequences)  # shuffle
        return dataset_input, dataset_target

    @staticmethod
    def _chunk_list(array, seq_length):
        return [array[i:i + seq_length] for i in range(0, len(array) - seq_length + 1, seq_length)]

    @staticmethod
    def _split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text
