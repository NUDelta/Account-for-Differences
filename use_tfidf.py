import pickle
import re
import numpy as np
from nltk.stem import PorterStemmer


def preprocessor(text):
    pattern = r"[^A-Za-z\s'-]+"
    return re.sub(pattern, "", text)

def stemmer(text):
    ps = PorterStemmer()
    return ps.stem(text)

def read_data(flag='state'):
    loaded = np.load('tfidf/%s-meta.npz' % flag, allow_pickle=True)
    paths = loaded['paths']
    vocabs = loaded['vocabulary']

    pathToIndex = {}
    wordToIndex = {}

    for idx, path in enumerate(paths):
        pathToIndex[path] = idx

    for idx, word in enumerate(vocabs):
        wordToIndex[word] = idx

    with open('tfidf/%s.mtx' % flag, 'rb') as f:
        matrix = pickle.load(f)
        matrix = matrix.todense()

    return pathToIndex, wordToIndex, matrix


sp, sw, sm = read_data('state')
rp, rw, rm = read_data('rating')
cp, cw, cm = read_data('city')


def cat2doc(category, flag='state', city=None):
    """Sushi Bars -> Sushi Bars.txt
    From Category to Document filepath """

    path = "reviewtext/%s/%s" % (flag, category)

    if flag == 'city' and city is not None:
        # some city name somehow contains slashes for example Wayne/Radnor in PA.
        path = path + '/' + city.replace('/', '-')

    return path + '.txt'


def get_score_(cat, word, catToIndex, wordToIndex, matrix, flag='state', city=None):
    path = cat2doc(cat, flag, city)
    word = stemmer(preprocessor(word))

    if path not in catToIndex:
        return -1
    if word not in wordToIndex:
        return -2

    x = catToIndex[path]
    y = wordToIndex[word]
    return matrix[x, y]


def get_score(cat, word, flag='state', city=None):
    if flag == 'state':
        return word + '|' + cat + ': ' + str(get_score_(cat, word, sp, sw, sm, flag))
    if flag == 'city':
        return word + '|' + city + ': ' + str(get_score_(cat, word, cp, cw, cm, flag, city))
    if flag == 'rating':
        return word + '|' + cat + ': ' + str(get_score_(cat, word, rp, rw, rm, flag))


if __name__ == '__main__':
    print(get_score('IL', 'beach'))
    print(get_score('CA', 'beach'))

    print(get_score('1', 'terrible', 'rating'))
    print(get_score('2', 'terrible', 'rating'))

    print(get_score('PA', 'pizza', 'city', 'Philadelphia'))
    print(get_score('CA', 'pizza', 'city', 'Goleta'))



