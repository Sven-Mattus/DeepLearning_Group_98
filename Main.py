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

    # generate dataset
    SEQ_LENGTH = 25
    BATCH_SIZE = 64

    # initialize network
    K = len(book_chars)
    lstm = LSTM(vocab_size=K, embedding_dim=256, nr_rnn_units=1024, batch_size=BATCH_SIZE)

    # train LSTM
    validation_set_len = BATCH_SIZE * SEQ_LENGTH * 2
    test_set_len = validation_set_len + BATCH_SIZE * SEQ_LENGTH * 2
    NR_EPOCHS = 1

    # filename
    learning_rate = 0.01 # needs to be adjusted in the LSTM class!!!
    layers = 1 # needs to be adjusted in the LSTM class!!!
    optimizer= 'GN' # needs to be adjusted in the LSTM class!!!
    temperature = 1.0 

    filename = f'{layers}''_layer_'f'{NR_EPOCHS}''_epoch_'f'{BATCH_SIZE}''_batchsize_'f'{learning_rate}''_eta_'f'{optimizer}'f'_optimizer_'f'{temperature}'f'_temperature'

    dataset_input, dataset_target = DataGenerator.create_array_dataset(book_as_ind[test_set_len:int(0.05*len(book_as_ind))],
                                                                       SEQ_LENGTH)  # arrays of size nr_seq x SEQ_LENGTH-1
    val_input, val_target = DataGenerator.create_array_dataset(book_as_ind[:validation_set_len],
                                                               SEQ_LENGTH)  # arrays of size nr_seq x SEQ_LENGTH-1
    test_input, test_target = DataGenerator.create_array_dataset(book_as_ind[validation_set_len:test_set_len],
                                                            SEQ_LENGTH)  # arrays of size nr_seq x SEQ_LENGTH-1

    history = lstm.train_network(dataset_input[len(dataset_input) % BATCH_SIZE:],
                                 dataset_target[len(dataset_target) % BATCH_SIZE:], NR_EPOCHS, BATCH_SIZE, val_input,
                                 val_target)
    
    # Evaluate the model
    test_loss, accuracy = lstm.evaluate(x=test_input, y=test_target, bs= BATCH_SIZE)
    gen_text = lstm.generate_text(temperature, start_string=" ", data_converter=data_converter)
    
    print("Test loss:", test_loss, '\n', "Accuracy:", accuracy, '\n', gen_text )

    with open('results/'f'{filename}''.txt', 'a') as f:
        # Append the loss value followed by a newline character
        f.write(f'{filename}'+'\n'+ "Test loss:" + str(test_loss) + '\n' + "Accuracy:" + str(accuracy) + '\n' + "Generated Text:" + str(gen_text)+ '\n')

    Evaluator.plot_history_loss(history, filename)



    # dataset = DataGenerator.create_tf_dataset(book_as_ind[validation_set_len: len(book_as_ind)], SEQ_LENGTH)
    # dataset_val = DataGenerator.create_tf_dataset(book_as_ind[:validation_set_len], SEQ_LENGTH)
    # history = lstm.train_network_with_tf_dataset(dataset, NR_EPOCHS, dataset_val)
