from DataConverter import DataConverter
from DataLoader import DataLoader

# load data
book = DataLoader.load_data()
book_chars = sorted(set(book))
data_converter = DataConverter(book_chars)
book_as_ind = data_converter.chars_to_ind(book)

# chunk in sequences
seq_length = 100
nr_seqs_per_epochs = len(book) // seq_length  # floor division
sequences = data_converter.chunk_list(book_as_ind, seq_length)
assert len(sequences) == nr_seqs_per_epochs
dataset = [(seq[:-1], seq[1:]) for seq in sequences]  # split into input and target text

