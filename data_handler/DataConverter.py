import numpy as np
import tensorflow as tf

class DataConverter:

    def __init__(self, book_chars):
        self._book_chars = book_chars
        self._char_to_ind, self._ind_to_char = self._generate_convert_dicts(book_chars)

    def chars_to_ind(self, chars):
        ind = np.zeros(len(chars), int)
        for i in range(len(chars)):
            ind[i] = int(self._char_to_ind[chars[i]])
        return ind

    def _generate_convert_dicts(self, book_chars):
        K = len(book_chars)
        positions = np.arange(0, K)
        char_to_ind = dict(zip(book_chars, positions))
        ind_to_char = dict(zip(positions, book_chars))
        return char_to_ind, ind_to_char

    @staticmethod
    def chunk_list(array, seq_length):
        return [array[i:i + seq_length] for i in range(0, len(array) - seq_length + 1, seq_length)]

    def split_input_target(self, chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    def chunk_list_of_tuples(self, dataset, batch_size):
        num_batches = len(dataset) // batch_size
        chunked_list = []
        for i in range(num_batches):
            batch = tuple(np.vstack(arrays) for arrays in zip(*dataset[i * batch_size:(i + 1) * batch_size]))
            chunked_list.append(batch)
        return chunked_list

    def create_tf_dataset(self, book_as_ind, seq_length):
        char_dataset = tf.data.Dataset.from_tensor_slices(book_as_ind)
        sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
        dataset = sequences.map(self.split_input_target)
        BATCH_SIZE = 64
        # Buffer size to shuffle the dataset (TF data is designed to work
        # with possibly infinite sequences, so it doesn't attempt to shuffle
        # the entire sequence in memory. Instead, it maintains a buffer in
        # which it shuffles elements).
        BUFFER_SIZE = 10000
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        return dataset

    def create_dataset(self, book_as_ind, seq_length):
        nr_seqs_per_epochs = len(book_as_ind) // seq_length  # floor division
        sequences = DataConverter.chunk_list(book_as_ind, seq_length)
        assert len(sequences) == nr_seqs_per_epochs
        data_block = np.vstack(sequences)
        dataset_input = data_block[:, :seq_length-1]
        dataset_target = data_block[:, 1:]
        # np.random.shuffle(sequences)  # shuffle
        return dataset_input, dataset_target

    def split_input_target(self, chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text


