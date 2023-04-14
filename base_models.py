# Adapted from
# https://github.com/youralien/affinder-search/blob/4dd427c98c98533b0550c3d484fba55d75f39818/yelp_academic_etl_training.py
# Author youralien

import os
import re
import io
import numpy as np
import pickle
import pymysql.cursors
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from nltk.stem import PorterStemmer

def all_states():
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 port=3306,
                                 password='suhuai2001',
                                 db='yelp',
                                 cursorclass=pymysql.cursors.Cursor)
    cursor = connection.cursor()
    query = "SELECT DISTINCT business.State FROM business"
    cursor.execute(query)
    states = [state[0] for state in cursor]
    '''
    os.mkdir('reviewtext/state')
    os.mkdir('reviewtext/city')
    for state in states:
        path1 = 'reviewtext/state' + '/' + state
        path2 = 'reviewtext/city' + '/' + state
        os.mkdir(path1)
        os.mkdir(path2)
    '''
    cursor.close()
    connection.close()
    return states
def get_cities():
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 port=3306,
                                 password='suhuai2001',
                                 db='yelp',
                                 cursorclass=pymysql.cursors.Cursor)

    cursor = connection.cursor()
    cities = {}
    states = all_states()
    query = ("SELECT DISTINCT business.City FROM business "
             "WHERE business.State=%s")

    for state in states:
        cursor.execute(query, state)
        cities[state] = [city[0] for city in cursor]
        '''
        for city in cities[state]:
            path = 'reviewtext/city' + '/' + state + '/' + city
            os.mkdir(path)
        '''
    cursor.close()
    connection.close()
    return cities

all_cities = get_cities()

def get_categories():
    categories_states = []
    categories_cities = []
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 port=3306,
                                 password='suhuai2001',
                                 db='yelp',
                                 cursorclass=pymysql.cursors.Cursor)

    cursor = connection.cursor()
    cities = {}
    states = all_states()
    query = ("SELECT DISTINCT business.City FROM business "
             "WHERE business.State=%s")






def stemmer(text):
    ps = PorterStemmer()

    return ps.stem(text).split()


stop_words = [stemmer(i)[0] for i in ENGLISH_STOP_WORDS]


def preprocessor(text):
    pattern = r"[^A-Za-z\s'-]+"
    return re.sub(pattern, "", text)


def cat2doc(state, cat, flag='state', city=None):
    """Sushi Bars -> Sushi Bars.txt
    From Category to Document filepath """

    path = "reviewtext/%s/%s" % (flag, state)

    if flag == 'city' and city is not None:
        # some city name somehow contains slashes for example Wayne/Radnor in PA.
        path = path + '/' + city.replace('/', '-')

    path = path + '/' + cat.replace('/', '-')
    return path + '.txt'


def cats2docs(state, categories, flag='state', city=None):

    if isinstance(categories, str):
        categories = (categories, )

    if flag == 'city':
        return [cat2doc(state, cat, 'city', city) for cat in categories]

    return [cat2doc(state, cat, flag) for cat in categories]


def write_document(cursor, state, cat, flag='state'):
    """ Given a yelp category, build out a text document
    which has all the reviews for that category """
    if flag == 'city':
        cities = all_cities[state]
        n_encoding_errors = 0
        n_review = 0
        for city in cities:
            query = ("SELECT review.Content "
                     "FROM review INNER JOIN business "
                     "ON review.B_id = business.Business_id "
                     "WHERE business.State=%s AND business.City=%s")
            print(query)
            cursor.execute(query, (state, city, cat))

            with io.open(cat2doc(state, cat, flag, city), 'w', encoding='utf8') as f:
                for text, in cursor:
                    try:
                        f.write(text)
                        f.write("\n")
                    except UnicodeEncodeError:
                        n_encoding_errors += 1
                    n_review += 1
        return n_encoding_errors, n_review

    query = ("SELECT review.Content "
             "FROM review "
             "INNER JOIN (business "
             "INNER JOIN category "
             "ON business.Business_id = category.Business_id) "
             "ON review.B_id = business.Business_id "
             "WHERE business.State=%s category.Category_name=%s")

    cursor.execute(query, (state, cat))

    n_encoding_errors = 0
    n_review = 0
    with io.open(cat2doc(state, cat, flag), 'w', encoding='utf8') as f:
        for text, in cursor:
            try:
                f.write(text)
                f.write("\n")
            except UnicodeEncodeError:
                n_encoding_errors += 1
            n_review += 1
    return n_encoding_errors, n_review


def document_text_iterator(states, flag='state'):
    for filepath in cats2docs(states,flag):
        with io.open(filepath, 'r', encoding='utf8') as f:
            yield f.read()


def document_iterator(states, flag='state'):
    for filepath in cats2docs(states, flag):
        yield filepath


def sql2txt(states, flag='state'):
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 port=3306,
                                 password='suhuai2001',
                                 db='yelp',
                                 cursorclass=pymysql.cursors.Cursor)

    cursor = connection.cursor()

    if isinstance(states, str):
        states = (states, )

    for state in states:
            n_errors, n_total = write_document(cursor, state, cat, flag)
            print("%s: %d errors, %d total" % (state, n_errors, n_total))

    cursor.close()
    connection.close()


def create_all_documents(flag='state'):
    states = all_states()
    # print("Creating %d documents" % len(cats))
    sql2txt(states, flag)


def vectorize_sklearn(states, flag='state'):
    # should I use the vocabulary from something like fasttext?
    vect = TfidfVectorizer(input='filename', preprocessor=preprocessor, tokenizer=stemmer,
                           vocabulary=None, token_pattern=None, stop_words=stop_words)
    X = vect.fit_transform(document_iterator(states, flag))
    vocabulary = vect.get_feature_names_out()
    return X, states, vocabulary


def save_pickle(matrix, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(matrix, outfile, pickle.HIGHEST_PROTOCOL)


def compute_and_save(flag='state'):
    states = all_states()
    X, states, vocabulary = vectorize_sklearn(states, flag)
    save_pickle(X, 'tfidf/%s.mtx' % flag)
    np.savez_compressed('tfidf/%s-meta' % flag,
                        paths=cats2docs(states, flag), vocabulary=vocabulary)


if __name__ == '__main__':
    # categories = ('Sushi Bars',
    #               'Bikes',
    #               'Dance Clubs')

    #create_all_documents(flag='state')
    # create_all_documents(flag='city')
    #compute_and_save('state')
    #compute_and_save('city')
    print('Y')

'''
    categories = all_categories()
    X, categories, vocabulary = vectorize_sklearn(categories)

    save_pickle(X, 'tfidf/state.mtx')
    np.savez_compressed('tfidf/state-meta',
                        paths=cats2docs(categories), vocabulary=vocabulary)
'''