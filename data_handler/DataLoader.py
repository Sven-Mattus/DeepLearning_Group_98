
import os


class DataLoader:

    relative_path_book = "/data/goblet_book.txt"

    @staticmethod
    def load_data():
        book = DataLoader._load_book()
        return book
        # todo: preprocess data? (e.g. all lower case)

    @staticmethod
    def _load_book():
        path = os.getcwd() + DataLoader.relative_path_book
        file = open(path, "r")
        book = file.read()
        file.close()
        return book
