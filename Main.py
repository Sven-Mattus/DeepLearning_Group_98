
import numpy as np
from data_handler.DataConverter import DataConverter
from data_handler.DataLoader import DataLoader

if __name__ == "__main__":
    # load data
    book = DataLoader.load_data()
    book_chars = sorted(set(book))
    data_converter = DataConverter(book_chars)
    book_as_ind = data_converter.chars_to_ind(book)

    # chunk text into sequences
    SEQ_LENGTH = 100
    nr_seqs_per_epochs = len(book_as_ind) // SEQ_LENGTH  # floor division
    sequences = data_converter.chunk_list(book_as_ind, SEQ_LENGTH)
    assert len(sequences) == nr_seqs_per_epochs
    dataset = [(seq[:-1], seq[1:]) for seq in sequences]  # split into input and target text: list of tuples

    # split sequences into batches
    BATCH_SIZE = 64
    np.random.shuffle(dataset)  # shuffle
    batched_dataset = data_converter.chunk_list_of_tuples(dataset, BATCH_SIZE)  # list of tuples that contain arrays of size batch_size x seq_length-1

    # set hyperparameters

    # initialize parameters

    # train LSTM

    # evaluate LSTM

    print()

