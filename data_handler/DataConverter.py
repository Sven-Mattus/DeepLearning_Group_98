import numpy as np


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

    def chunk_list(self, array, seq_length):
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
