import numpy as np


class DataConverter:

    def __init__(self, book_chars):
        self._book_chars = book_chars
        self._char_to_ind, self._ind_to_char = self._generate_convert_dicts(book_chars)

    def chars_to_ind(self, chars):
        """
        converts a string into 1D array of indices
        :param chars: string
        :return:    1D array of indices
        """
        ind = np.zeros(len(chars), int)
        for i in range(len(chars)):
            ind[i] = int(self._char_to_ind[chars[i]])
        return ind

    def ind_to_char(self, ind):
        """
        converts an index into corresponding char
        :param ind: string
        :return:    1D array of indices
        """
        return self._ind_to_char[ind]

    def _generate_convert_dicts(self, book_chars):
        K = len(book_chars)
        positions = np.arange(0, K)
        char_to_ind = dict(zip(book_chars, positions))
        ind_to_char = dict(zip(positions, book_chars))
        return char_to_ind, ind_to_char




