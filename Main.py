
import numpy as np
from data_handler.DataConverter import DataConverter
from data_handler.DataLoader import DataLoader
from neural_network.LSTM import LSTM
from Translated_Matlab_Code.train_network_RNN import TrainNetwork
from Translated_Matlab_Code.VanillaRNN_Class import VanillaRNN
from Translated_Matlab_Code import forward_pass as fp
from evaluation.plot_loss import plot_loss
from evaluation.synthesize_text import synthesize_text, print_synthesized_text
from evaluation.evaluate_loaded_weights import evaluate_loaded_weights

if __name__ == "__main__":
    # load data
    book = DataLoader.load_data()
    book_chars = sorted(set(book))
    data_converter = DataConverter(book_chars)
    book_as_ind = data_converter.chars_to_ind(book)

    # chunk text into sequences
    SEQ_LENGTH = 25
    nr_seqs_per_epochs = len(book_as_ind) // SEQ_LENGTH  # floor division
    sequences = data_converter.chunk_list(book_as_ind, SEQ_LENGTH)
    assert len(sequences) == nr_seqs_per_epochs
    dataset = [(seq[:-1], seq[1:]) for seq in sequences]  # split into input and target text: list of tuples

    # split sequences into batches
    BATCH_SIZE = 64
    np.random.shuffle(dataset)  # shuffle
    batched_dataset = data_converter.chunk_list_of_tuples(dataset, BATCH_SIZE)  # list of tuples that contain arrays of size batch_size x seq_length-1

    # # initialize network
    # ETA = 0.01
    # K = len(book_chars)
    # lstm = LSTM(input_size=K, output_size=len(book_chars), learning_rate=ETA)

    # # train LSTM
    # NR_EPOCHS = 2
    # lstm.train_network(batched_dataset, nr_epochs=2)
    # # evaluate LSTM

    #Train the RNN
    # convert the book chars back to ind
    sig = .01 # sigma for random distribution
    ETA = 0.01 #learning rate
    K = len(book_chars)
    m = 100 # dimensionality of hidden state

    RNN = VanillaRNN(sig, m, K)
    RNN_loaded = VanillaRNN(sig, m, K)

    #RNN_trained, smooth_loss, smooth_loss_val = TrainNetwork(book, 100001, SEQ_LENGTH, RNN, ETA, data_converter)

    evaluate_loaded_weights(SEQ_LENGTH, book_as_ind, data_converter, m)

    # print(loss_loaded/SEQ_LENGTH)

    #plot_loss(smooth_loss, smooth_loss_val)

