import numpy as np


class DataConverter:

    def __init__(self, book_chars):
        self._book_chars = book_chars
        self._char_to_ind, self._ind_to_char = self._generate_convert_dicts(book_chars)
        self.K = (len(self._char_to_ind), )

    def chars_to_ind(self, chars):
        ind = np.zeros(len(chars), int)
        for i in range(len(chars)):
            ind[i] = int(self._char_to_ind[chars[i]])
        return ind
    
    def ind_to_chars(self, indices):
        chars = []
        for i in range(len(indices)):
            chars.append(self._ind_to_char(indices[i]))
        return chars

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
    
    def one_hot_encode(self, chars):
        one_hot = np.zeros((self.K, chars.shape[1]))
        for i in range(len(chars)):
            ind = self.char_to_ind(chars(i))
            one_hot[ind, i] = 1
        return one_hot
    
    def one_hot_to_chars(self, one_hot):
        indices = np.argmax(one_hot, axis=0)
        chars = self.ind_to_chars(indices)
        return chars