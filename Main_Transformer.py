import numpy as np

from data_handler.DataConverter import DataConverter
from data_handler.DataLoader import DataLoader
from data_handler.DatasetGenerator import DataGenerator
from neural_network.Transformer import Transformer
from evaluation.Evaluator import Evaluator

text = DataLoader.load_data()

vocab = sorted(set(text))
vocab_size = len(vocab)
data_converter = DataConverter(vocab)
book_as_ind = np.array(data_converter.chars_to_ind(text))

# Create training examples / targets
SEQ_LENGTH = 80
BATCH_SIZE = 100
validation_set_len = BATCH_SIZE * SEQ_LENGTH * 5
test_set_len = validation_set_len + BATCH_SIZE * SEQ_LENGTH * 5
examples_per_epoch = len(text) // (SEQ_LENGTH + 1)

# dataset = DataGenerator.create_tf_dataset(book_as_ind[validation_set_len: len(book_as_ind)], seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE)
# dataset_val = DataGenerator.create_tf_dataset(book_as_ind[:validation_set_len], SEQ_LENGTH, BATCH_SIZE)

dataset_input, dataset_target = DataGenerator.create_array_dataset(book_as_ind[test_set_len:],
                                                                   SEQ_LENGTH)  # arrays of size nr_seq x SEQ_LENGTH-1
val_input, val_target = DataGenerator.create_array_dataset(book_as_ind[:validation_set_len],
                                                           SEQ_LENGTH)  # arrays of size nr_seq x SEQ_LE
test_input, test_target = DataGenerator.create_array_dataset(book_as_ind[validation_set_len:test_set_len],
                                                                 SEQ_LENGTH)  # arrays of size nr_seq x SEQ_LENGTH-1

# train
num_layers = 1
num_heads = 6
embedding_dim = 512
dropout_rate = 0.001
learning_rate = 0.001
temperature = 1.0
nr_epochs = 20

transformer = Transformer(vocab_size=vocab_size, num_layers=num_layers, embedding_dim=embedding_dim,
                          num_heads=num_heads, batch_size=BATCH_SIZE, learning_rate=learning_rate,
                          dropout_rate=dropout_rate, seq_length=SEQ_LENGTH)
print(transformer.generate_text(temperature=temperature, start_string=' ', data_converter=data_converter))
history = transformer.train_network(dataset_input=dataset_input, dataset_target=dataset_target, nr_epochs=nr_epochs,
                          val_input=val_input, val_target=val_target, batch_size=BATCH_SIZE)
gen_text = transformer.generate_text(temperature=temperature, start_string=' ', data_converter=data_converter)
print(gen_text)

filename = f'{num_layers}''_lay_'f'{num_heads}''_head_'f'{nr_epochs}''_epo_'f'{BATCH_SIZE}''_batchs_'f'{learning_rate}''_eta_'f'{dropout_rate}'f'_drop_'f'{temperature}'f'_temp_'f'{embedding_dim}'f'_embeddim_'f'{SEQ_LENGTH}'f'seql'

test_loss, accuracy = transformer.evaluate(x=test_input, y=test_target, bs=BATCH_SIZE)

#transformer.save_weights(filename)

with open('results/'f'{filename}''.txt', 'a') as f:
    # Append the loss value followed by a newline character
    f.write(f'{filename}' + '\n' + "Test loss:" + str(test_loss) + '\n' + "Accuracy:" + str(
        accuracy) + '\n' + "Generated Text:" + str(gen_text) + '\n')
    
Evaluator.plot_history_loss(history, filename)