from googletrans import Translator
from DataLoader import DataLoader


def translate_text(text, src_lang='en', dest_lang='fr'):
    translator = Translator()
    translated = translator.translate(text, src=src_lang, dest=dest_lang).text
    back_translated = translator.translate(translated, src=dest_lang, dest=src_lang).text
    return back_translated


def augment_dataset(book_text, segment_length=5000):
    augmented_text = ""
    for i in range(0, len(book_text), segment_length):
        segment = book_text[i:i+segment_length]
        augmented_segment = translate_text(segment)
        augmented_text += augmented_segment
    return augmented_text


book = DataLoader.load_data()

# Get the augmented book and save it as new book
augmented_book = augment_dataset(book)
with open("data/french_goblet_book.txt", "w") as file:
    file.write(augmented_book)