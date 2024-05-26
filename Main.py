from Initializer import Initializer
from data_handler.DataConverter import DataConverter
from data_handler.DataLoader import DataLoader
from data_handler.DatasetGenerator import DataGenerator
from evaluation.Evaluator import Evaluator
from neural_network.LSTM import LSTM

if __name__ == "__main__":
    # load data
    book = DataLoader.load_data()
    book_chars = sorted(set(book))
    data_converter = DataConverter(book_chars)
    book_as_ind = data_converter.chars_to_ind(book)

    ### Enter here the script:
    ### Set the parameters
    ### as well as change the optimizer; GlorotNormal or GlorotUniform 
    ### and the learning rate

    # set parameters
    layers = 2                          # default = 1
    optimizer = Initializer.GN.value    # default = GN
    learning_rate = 0.01                # default = 0.01
    nr_rnn_units = 1024                 # default = 1024
    temperature = 1.0                   # default = 1.0

    # generate dataset
    SEQ_LENGTH = 25                     # default = 25
    BATCH_SIZE = 64                     # default = 64

    # initialize network
    K = len(book_chars)
    lstm = LSTM(vocab_size=K, embedding_dim=256, nr_rnn_units=nr_rnn_units, batch_size=BATCH_SIZE, nr_layers=layers,
                learning_rate=learning_rate, initializer=optimizer)

    # train LSTM
    validation_set_len = BATCH_SIZE * SEQ_LENGTH * 20
    test_set_len = validation_set_len + BATCH_SIZE * SEQ_LENGTH * 20
    NR_EPOCHS = 10

    filename = f'{layers}''_lay_'f'{NR_EPOCHS}''_epo_'f'{BATCH_SIZE}''_batchs_'f'{learning_rate}''_eta_'f'{optimizer}'f'_opti_'f'{temperature}'f'_temp_'f'{nr_rnn_units}'f'_units_'f'{SEQ_LENGTH}'f'seql''_p0.8'

    dataset_input, dataset_target = DataGenerator.create_array_dataset(book_as_ind[test_set_len:],
                                                                       SEQ_LENGTH)  # arrays of size nr_seq x SEQ_LENGTH-1
    val_input, val_target = DataGenerator.create_array_dataset(book_as_ind[:validation_set_len],
                                                               SEQ_LENGTH)  # arrays of size nr_seq x SEQ_LENGTH-1
    test_input, test_target = DataGenerator.create_array_dataset(book_as_ind[validation_set_len:test_set_len],
                                                                 SEQ_LENGTH)  # arrays of size nr_seq x SEQ_LENGTH-1

    history = lstm.train_network(dataset_input[len(dataset_input) % BATCH_SIZE:],
                                 dataset_target[len(dataset_target) % BATCH_SIZE:], NR_EPOCHS, BATCH_SIZE, val_input,
                                 val_target)

    lstm.save_weights(filename)

    # Evaluate the model
    test_loss, accuracy = lstm.evaluate(x=test_input, y=test_target, bs=BATCH_SIZE)
    gen_text = lstm.generate_text(temperature, start_string=" ", data_converter=data_converter)

    print("Test loss:", test_loss, '\n', "Accuracy:", accuracy, '\n', gen_text)

    with open('results/'f'{filename}''.txt', 'a') as f:
        # Append the loss value followed by a newline character
        f.write(f'{filename}' + '\n' + "Test loss:" + str(test_loss) + '\n' + "Accuracy:" + str(
            accuracy) + '\n' + "Generated Text:" + str(gen_text) + '\n')

    Evaluator.plot_history_loss(history, filename)
