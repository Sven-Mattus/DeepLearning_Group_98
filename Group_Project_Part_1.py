from DataConverter import DataConverter
from DataLoader import DataLoader

book = DataLoader.load_data()
book_chars = sorted(set(book))
data_converter = DataConverter(book_chars)

