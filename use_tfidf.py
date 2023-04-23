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
    loaded = np.load('tfidf/state-meta.npz', allow_pickle=True)
    states = loaded['states']
    return states


def load_cities():
    """

    Returns dict[state]: a list of cities in that state
    -------

    """
    loaded = np.load('tfidf/city-meta.npz', allow_pickle=True)
    all_cities = loaded['all_cities']
    return all_cities


def load_categories():
    """
    Returns
    dict[state]: a list of categories of that state
    dict[state][city]: a list of categories of that city
    -------

    """
    loaded = np.load('tfidf/category-meta.npz', allow_pickle=True)
    return loaded['categories_of_state'], loaded['categories_of_city']


def cat2doc(state, cat, flag='state', city=None):
    """CA, Goleta, Sushi Bars -> reviewtext/city/CA/Goleta/SushiBars.txt
    From state,cat,(city) to Document filepath """

    path = "reviewtext/%s/%s" % (flag, state)

    if flag == 'city' and city is not None:
        # some city name somehow contains slashes for example Wayne/Radnor in PA.
        path = path + '/' + city.replace('/', '-')

    path = path + '/' + cat.replace('/', '-')
    return path + '.txt'



def read_data(flag='state'):
    """
    given a state (and a city), load the related tf-idf matrix, filepaths and vocabulary
    where the filepaths are the row names and vocabulary contains the row names of the tf-idf matrix

    Output a tf-idf matrix, a dictionary that maps each filepath to the row index, and a dictionary that maps each
    word to the column index

    """

    with open('tfidf/%s.mtx' % flag, 'rb') as f:
        matrix = pickle.load(f)
        # matrix = matrix.todense()

    loaded = np.load('tfidf/%s-features.npz' % flag, allow_pickle=True)

    # categories is a list of column names of the tf-idf matrix
    categories = loaded['document_names']

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


catToIndex_state, wordToIndex_state, matrix_state = read_data('state')
catToIndex_city, wordToIndex_city, matrix_city = read_data('city')


def retrieve_score_(filepath, words, catToIndex, wordToIndex, matrix):
    """
    retrieve the tf-idf score of a word in relation to a category when the matrix and mappings from names to indices are given
    """

    phrase_score = 0.0
    words = stemmer(preprocessor(words))
    word_list = re.findall(r"[A-Za-z'-]+", words)

    if filepath not in catToIndex:
        print("filepath doesn't exist: "+filepath)
        return -1

    x = catToIndex[filepath]

    for word in word_list:
        if word not in wordToIndex:
            print("word doesn't exist: "+word)
            return -2

        y = wordToIndex[word]
        phrase_score += matrix[x, y]
    return phrase_score


def retrieve_score(words, cat, state, flag='state', city=None):

    """
    Retrieve the tf-idf score of a word in relation to a category in a setting

    The preprocessor is necessary in order to match the precessing and tokenizing steps when calculating
    tf-idf score in base_models.py
    """
    filepath = cat2doc(state, cat, flag, city)

    if flag == 'state':
        return retrieve_score_(filepath, words, catToIndex_state, wordToIndex_state, matrix_state)

    if flag == 'city':
        return retrieve_score_(filepath, words, catToIndex_city, wordToIndex_city, matrix_city)


def get_top_k(word, k, state, flag='state', city=None):
    """
    get top k related categories for a given word and a given state or city
    """
    scores = []
    catToIndex, wordToIndex, matrix = read_data(flag)
    for cat in catToIndex.keys():
        score = retrieve_score_(cat, word, catToIndex, wordToIndex, matrix, flag, city)
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

    #print(retrieve_score('fun', 'Restaurants', 'AZ'))

    words = 'places to exhaust children'

    print(retrieve_score(words, 'Restaurants', 'AZ'))
    print(retrieve_score(words, 'Restaurants', 'CA', 'city', 'Goleta'))


'''
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

        # input('press enter to continue')
'''


