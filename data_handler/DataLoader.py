
import os


class DataLoader:

    relative_path_book = "/data/goblet_book.txt"

    @staticmethod
    def load_data():
        book = DataLoader.load_book()
        return book
        # todo: chunk into training, validation and test data
        # todo: preprocess data? (e.g. all lower case)
        # return x_train, y_train, x_validation, y_validation, x_test, y_test

    @staticmethod
    def load_book():
        path = os.getcwd() + DataLoader.relative_path_book
        file = open(path, "r")
        book = file.read()
        file.close()
        return book
