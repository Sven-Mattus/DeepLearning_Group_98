import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
import json

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def load_config():
    """Loads configuration from a JSON file."""
    with open('data_handler/config.json', 'r') as file:
        config = json.load(file)
    return config

def apply_augmentations(book_text):
    """Applies text augmentations based on the configuration provided."""
    config = load_config()

    sentences = text_to_sentences(book_text)
    text_synonym, text_deletion, text_insertion, text_swap, text_deletion, book_shuffled, book_unique = '', '', '', '', '', '', ''
    
    for sentence in sentences:
        if config['augmentations']['synonym_replacement']['enabled']:
            augmented_text = synonym_replacement(sentence, config['augmentations']['synonym_replacement']['changed_words'])
            text_synonym += augmented_text
        if config['augmentations']['random_insertion']['enabled']:
            augmented_text = random_insertion(sentence, config['augmentations']['random_insertion']['inserted_words'])
            text_insertion += augmented_text
        if config['augmentations']['random_swap']['enabled']:
            augmented_text = random_swap(sentence, config['augmentations']['random_swap']['amount_swaps'])
            text_swap += augmented_text
        if config['augmentations']['random_deletion']['enabled']:
            augmented_text = random_deletion(sentence, config['augmentations']['random_deletion']['deletion_probability'])
            text_deletion += augmented_text
    
    if config['augmentations']['shuffle_sentences']['enabled']:
        augmented_text = shuffle_sentences(book_text)
        book_shuffled += augmented_text
    if config['augmentations']['exclude_duplicates']['enabled']:
        augmented_text = exclude_duplicate_sentences(book_text)
        book_unique += augmented_text

    new_book_text = text_synonym + text_insertion + text_swap + text_deletion + book_shuffled + book_unique
    
    return new_book_text

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacement(sentence, n):
    words = word_tokenize(sentence)
    eligible_words = [word for word in words if word not in stop_words and word.isalnum()]
    random_words = random.sample(eligible_words, min(n, len(eligible_words)))
    new_sentence = sentence
    for word in random_words:
        synonyms = get_synonyms(word)
        if synonyms:
            new_word = random.choice(synonyms)
            new_sentence = new_sentence.replace(word, new_word, 1)
    return new_sentence

def random_insertion(sentence, n):
    words = word_tokenize(sentence)
    for _ in range(n):
        synonyms = []
        while not synonyms:
            word = random.choice(words)
            synonyms = get_synonyms(word)
        synonym = random.choice(synonyms)
        position = random.randint(0, len(words))
        words.insert(position, synonym)
    return ' '.join(words)

def random_swap(sentence, n):
    words = word_tokenize(sentence)
    length = len(words)
    for _ in range(n):
        idx1, idx2 = random.sample(range(length), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

def random_deletion(sentence, p):
    words = word_tokenize(sentence)
    new_words = [word for word in words if random.random() > p]
    return ' '.join(new_words)

def text_to_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

def shuffle_sentences(text):
    """Shuffles the sentences in the given text to create a new sample."""
    sentences = sent_tokenize(text)
    random.shuffle(sentences)
    return ' '.join(sentences)

def exclude_duplicate_sentences(text):
    """Removes duplicate sentences from the text to create a new sample."""
    sentences = sent_tokenize(text)
    unique_sentences = list(dict.fromkeys(sentences))  # Removes duplicates while preserving order
    return ' '.join(unique_sentences)


"""
# Example Usage
text = "Sentence1. Sentence2. Sentence3. Sentence3. Sentence4. Sentence4."

shuffled_text = shuffle_sentences(text)
print("Shuffled Text:", shuffled_text)

unique_text = exclude_duplicate_sentences(text)
print("Unique Text:", unique_text)


sentence = "This article will focus on summarizing data augmentation techniques in NLP."
print("Original:", sentence)
print("Synonym Replacement:", synonym_replacement(sentence, 2))
print("Random Insertion:", random_insertion(sentence, 2))
print("Random Swap:", random_swap(sentence, 2))
print("Random Deletion:", random_deletion(sentence, 0.25))

large_text = "Chapter 1. Once upon a time, there was an old library full of books. Each book held a different tale. \
Chapter 2. The library was located in a small village, surrounded by a lush forest. The villagers often visited the library to read and learn."
sentences = text_to_sentences(large_text)
for sentence in sentences:
    print(sentence)
"""