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
SEQ_LENGTH = 28
BATCH_SIZE = 64
validation_set_len = BATCH_SIZE * SEQ_LENGTH * 2
examples_per_epoch = len(text) // (SEQ_LENGTH + 1)

dataset = DataGenerator.create_tf_dataset(book_as_ind[validation_set_len: len(book_as_ind)], SEQ_LENGTH)
dataset_val = DataGenerator.create_tf_dataset(book_as_ind[:validation_set_len], SEQ_LENGTH)

# train
num_layers = 1  # 4
embedding_dim = 256
num_heads = 1  # 8
dropout_rate = 0.00
learning_rate = 0.01

transformer = Transformer(vocab_size=vocab_size, num_layers=num_layers, embedding_dim=embedding_dim,
                          num_heads=num_heads, batch_size=BATCH_SIZE, learning_rate=learning_rate,
                          dropout_rate=dropout_rate, seq_length=SEQ_LENGTH)

transformer.train_network_with_tf_dataset(dataset, nr_epochs=5, dataset_val=dataset_val)
