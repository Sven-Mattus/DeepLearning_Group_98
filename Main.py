import numpy as np

from data_handler.DataConverter import DataConverter
from data_handler.DataLoader import DataLoader
from neural_network.LSTM import LSTM

if __name__ == "__main__":
    # load data
    book = DataLoader.load_data()
    book_chars = sorted(set(book))
    data_converter = DataConverter(book_chars)
    book_as_ind = data_converter.chars_to_ind(book)

    # generate dataset
    SEQ_LENGTH = 100
    BATCH_SIZE = 64

    # initialize network
    K = len(book_chars)
    lstm = LSTM(vocab_size=K, embedding_dim=256, nr_rnn_units=1024, batch_size=BATCH_SIZE)

    # train LSTM
    NR_EPOCHS = 20
    dataset_input, dataset_target = data_converter.create_dataset(book_as_ind, SEQ_LENGTH)  # arrays of size nr_seq x SEQ_LENGTH-1
    history = lstm.train_network(dataset_input[3:], dataset_target[3:], NR_EPOCHS, BATCH_SIZE)
    # dataset = data_converter.create_tf_dataset(book_as_ind, SEQ_LENGTH)
    # history = lstm.train_network_with_tf_dataset(dataset, NR_EPOCHS)
    lstm.render_training_history(history)

