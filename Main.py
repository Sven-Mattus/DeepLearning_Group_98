#load standard python libraries

import math
import numpy as np
import pandas as pd

# import own implementations of relevant models
from LSTM import *
from VanillaRNN import *


if __name__ == "__main__":
    
    #relative paths to the data
    text_file = r'goblet_book.txt'

    # Reading the text file
    with open(text_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # get only the unique charachters in oder of appearance
    characters_list = list(set(text))

    # check the ascii code for special characters
    # for i in range(len(characters_list)):
    #     print(f'Character: {characters_list[i]}, ASCII Code: {ord(characters_list[i])}. \n')

    K = len(characters_list)

    # create a set of key and value sequences
    # to map keys and values to each other, use a python dict
    position = np.arange(1,K)

    # initialize vectors for one-hot encoding
    # for char_to_ind keys are unique chars and values is position of appearance
    # ind_to_char works vice versa
    char_to_ind = dict(zip(characters_list, position))
    ind_to_char = dict(zip(position, characters_list))

    test = 0