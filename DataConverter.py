import numpy as np


class DataConverter:

    def __init__(self, book_chars):
        self._book_chars = book_chars
        self._char_to_ind, self._ind_to_char = self._generate_convert_dicts(book_chars)

    def chars_to_ind(self, chars):
        ind = np.zeros(len(chars))
        for i in range(len(chars)):
            ind[i] = self._char_to_ind[chars[i]]
        return ind

    def _generate_convert_dicts(self, book_chars):
        K = len(book_chars)
        char_to_ind = {}
        ind_to_char = {}
        for i in range(K):
            char_to_ind[book_chars[i]] = i
            ind_to_char[i] = book_chars[i]
        return char_to_ind, ind_to_char

    def chunk_list(self, array, seq_length):
        return [array[i:i + seq_length] for i in range(0, len(array) - seq_length + 1, seq_length)]

    def split_input_target(self, chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text
