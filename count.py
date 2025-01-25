from nltk.tokenize import word_tokenize
import string

def word_count(text, task):
    if task == 'cnndm':
        return len(text.strip().split())
    else:
        return len([word for word in word_tokenize(text) if word not in string.punctuation])