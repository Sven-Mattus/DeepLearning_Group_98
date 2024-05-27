import numpy as np

from data_handler.DataConverter import DataConverter
from data_handler.DataLoader import DataLoader
from data_handler.DatasetGenerator import DataGenerator
from neural_network.Transformer import Transformer

text = DataLoader.load_data()
text = text[:int(0.5 * len(text))]
vocab = sorted(set(text))
vocab_size = len(vocab)
data_converter = DataConverter(vocab)
book_as_ind = np.array(data_converter.chars_to_ind(text))

# Create training examples / targets
SEQ_LENGTH = 80
BATCH_SIZE = 100
validation_set_len = BATCH_SIZE * SEQ_LENGTH * 2
examples_per_epoch = len(text) // (SEQ_LENGTH + 1)

# dataset = DataGenerator.create_tf_dataset(book_as_ind[validation_set_len: len(book_as_ind)], seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE)
# dataset_val = DataGenerator.create_tf_dataset(book_as_ind[:validation_set_len], SEQ_LENGTH, BATCH_SIZE)

dataset_input, dataset_target = DataGenerator.create_array_dataset(book_as_ind[validation_set_len:],
                                                                   SEQ_LENGTH)  # arrays of size nr_seq x SEQ_LENGTH-1
val_input, val_target = DataGenerator.create_array_dataset(book_as_ind[:validation_set_len],
                                                           SEQ_LENGTH)  # arrays of size nr_seq x SEQ_LE

# train
num_layers = 1
num_heads = 1
embedding_dim = 256 * num_heads
dropout_rate = 0.001
learning_rate = 0.001
temperature = 1.0
nr_epochs = 10

transformer = Transformer(vocab_size=vocab_size, num_layers=num_layers, embedding_dim=embedding_dim,
                          num_heads=num_heads, batch_size=BATCH_SIZE, learning_rate=learning_rate,
                          dropout_rate=dropout_rate, seq_length=SEQ_LENGTH)
print(transformer.generate_text(temperature=temperature, start_string=' ', data_converter=data_converter))
transformer.train_network(dataset_input=dataset_input, dataset_target=dataset_target, nr_epochs=nr_epochs,
                          val_input=val_input, val_target=val_target, batch_size=BATCH_SIZE)
print(transformer.generate_text(temperature=temperature, start_string=' ', data_converter=data_converter))
