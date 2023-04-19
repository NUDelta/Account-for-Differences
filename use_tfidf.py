import pickle
import re
import numpy as np
from nltk.stem import PorterStemmer


def preprocessor(text):
    pattern = r"[^A-Za-z\s'-]+"
    return stemmer(re.sub(pattern, "", text))


def stemmer(text):
    ps = PorterStemmer()
    return ps.stem(text)


def load_states():
    """

    Returns a list of states in the dataset
    -------

    """
    loaded = np.load('tfidf/state-meta', allow_pickle=True)
    states = loaded['states']
    return states


def load_cities():
    """

    Returns dict[state]: a list of cities in that state
    -------

    """
    loaded = np.load('tfidf/city-meta', allow_pickle=True)
    all_cities = loaded['all_cities']
    return all_cities


def read_data(state, flag='state', city=None):
    """
    given a state (and a city), load the related tf-idf matrix, categories and vocabulary
    where the categories are the row names and vocabulary contains the row names of the tf-idf matrix

    Output a tf-idf matrix, a dictionary that maps each category name to the row index, and a dictionary that maps each
    word to the column index

    """
    matrix = None
    loaded = None

    if flag == 'state':
        with open('tfidf/%s/%s.mtx' % (flag, state), 'rb') as f:
            matrix = pickle.load(f)
            matrix = matrix.todense()

        loaded = np.load('tfidf/%s/%s-meta.npz' % (flag, state), allow_pickle=True)

    if flag == 'city':
        with open('tfidf/%s/%s/%s.mtx' % (flag, state, city), 'rb') as f:
            # the tf-idf matrix
            matrix = pickle.load(f)
            matrix = matrix.todense()

        loaded = np.load('tfidf/%s/%s/%s-meta.npz' % (flag, state, city), allow_pickle=True)

    # categories is a list of column names of the tf-idf matrix
    categories = loaded['categories']

    # vocabs is a list of row names of the tf-idf matrix
    vocabs = loaded['vocabulary']

    # reverse map names to indices so that we can easily retrieve a tf-idf score given a document and a word
    catToIndex = {}
    wordToIndex = {}

    for idx, category in enumerate(categories):
        catToIndex[category] = idx

    for idx, word in enumerate(vocabs):
        wordToIndex[word] = idx

    return catToIndex, wordToIndex, matrix


def retrieve_score_(word, cat, catToIndex, wordToIndex, matrix):
    """
    retrieve the tf-idf score of a word in relation to a category when the matrix and mappings from names to indices are given
    """

    word = stemmer(preprocessor(word))

    if cat not in catToIndex:
        return -1
    if word not in wordToIndex:
        return -2

    x = catToIndex[cat]
    y = wordToIndex[word]
    return matrix[x, y]


def retrieve_score(word, cat, state, flag='state', city=None):

    """
    Retrieve the tf-idf score of a word in relation to a category in a setting
    Don't use this function when retreving multiple scores. It is not efficient as it calls load_data every time

    The preprocessor is necessary in order to match the precessing and tokenizing steps when calculating
    tf-idf score in base_models.py
    """

    word = preprocessor(word)

    if flag == 'state':
        catToIndex, wordToIndex, matrix = read_data(state, flag)
        return retrieve_score_(word, cat, catToIndex, wordToIndex, matrix)

    elif flag == 'city':
        catToIndex, wordToIndex, matrix = read_data(state, flag, city)
        return retrieve_score_(word, cat, catToIndex, wordToIndex, matrix)

    else:
        print('unknown flag')


"""
get top k related categories for a given word and a given state or city
"""
def get_top_k(word, k, state, flag='state', city=None):
    scores = []
    catToIndex, wordToIndex, matrix = read_data(state, flag, city)
    for cat in catToIndex.keys():
        score = retrieve_score_(word, cat, catToIndex, wordToIndex, matrix)
        scores.append((cat, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    # store the top k categories with their scores to a csv file
    with open('top_k.csv', 'w') as f:
        for i in range(k):
            if i < len(scores):
                f.write('%s,%s\n' % (scores[i][0], scores[i][1]))

    return scores[:k]


if __name__ == '__main__':

    # c, w, m = read_data('AZ')
    # print(c.keys())

    # print(retrieve_score('fun', 'Restaurants', 'AZ'))

    case1 = [
        ('fun','CA'),
        ('fun','AZ'),
        ('fun','TX'),
        ('fun','IL'),
        ('fun','PA'),
        ('fun','MA'),
    ]

    case2 = [
        ('fat food','CA'),
        ('fat food','AZ'),
        ('fat food','TX'),
        ('fat food','IL'),
        ('fat food','PA'),
        ('fat food','MA'),
    ]

    case3 = [
        ('healthy food','CA'),
        ('healthy food','AZ'),
        ('healthy food','TX'),
        ('healthy food','IL'),
        ('healthy food','PA'),
        ('healthy food','MA'),
    ]

    case4 = [
        ('danger','CA'),
        ('danger','AZ'),
        ('danger','TX'),
        ('danger','IL'),
        ('danger','PA'),
        ('danger','MA'),
    ]

    case5 = [
        ('private','CA'),
        ('private','AZ'),
        ('private','TX'),
        ('private','IL'),
        ('private','PA'),
        ('private','MA'),  
    ]

    case6 = [
        ('food','CA'),
        ('food','AZ'),
        ('food','TX'),
        ('food','IL'),
        ('food','PA'),
        ('food','MA'),
    ]

    case7 = [
        ('relax','CA'),
        ('relax','AZ'),
        ('relax','TX'),
        ('relax','IL'),
        ('relax','PA'),
        ('relax','MA'),
    ]

    case8 = [
        ('fat','CA'),
        ('fat','AZ'),
        ('fat','TX'),
        ('fat','IL'),
        ('fat','PA'),
        ('fat','MA'),
    ]

    for case in case8:
        print('=====================')
        print(case)
        print(get_top_k(case[0], 10, case[1]))

        input('press enter to continue')



