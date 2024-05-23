import numpy as np
import tensorflow as tf

from data_handler.DataConverter import DataConverter
from data_handler.DataLoader import DataLoader
from data_handler.DatasetGenerator import DataGenerator
from neural_network.Transformer import Transformer

text = DataLoader.load_data()
text = text[:int(0.05 * len(text))]
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
num_layers = 4
embedding_dim = 256
num_heads = 8
dff = 512
dropout_rate = 0.0

transformer = Transformer(vocab_size, num_layers, embedding_dim, num_heads, dff, dropout_rate)

transformer.train_network_with_tf_dataset(dataset, nr_epochs=5, dataset_val=dataset_val)
