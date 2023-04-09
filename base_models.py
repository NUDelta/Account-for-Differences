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
import nltk
from nltk.stem import PorterStemmer


def get_cities():
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 port=3306,
                                 password='suhuai2001',
                                 db='yelp',
                                 cursorclass=pymysql.cursors.Cursor)

    cursor = connection.cursor()
    cities = {}
    categories = all_categories()
    query = ("SELECT DISTINCT business.City FROM business "
             "WHERE business.State=%s")

    for cat in categories:
        cursor.execute(query, cat)
        cities[cat] = [city[0] for city in cursor]

    return cities


def stemmer(text):
    ps = PorterStemmer()

    return ps.stem(text).split()


stop_words = [stemmer(i)[0] for i in ENGLISH_STOP_WORDS]


def preprocessor(text):
    pattern = r"[^A-Za-z\s'-]+"
    return re.sub(pattern, "", text)


def cat2doc(category, flag='state', city=None):
    """Sushi Bars -> Sushi Bars.txt
    From Category to Document filepath """

    path = "reviewtext/%s/%s" % (flag, category)

    if flag == 'city' and city is not None:
        # some city name somehow contains slashes for example Wayne/Radnor in PA.
        path = path + '/' + city.replace('/', '-')

    return path + '.txt'


def cats2docs(categories, flag):

    if isinstance(categories, str):
        categories = (categories, )

    if flag == 'city':
        return [cat2doc(cat, 'city', city) for cat in categories for city in all_cities[cat]]

    return [cat2doc(cat, flag) for cat in categories]


def all_categories(flag='state'):
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 port=3306,
                                 password='suhuai2001',
                                 db='yelp',
                                 cursorclass=pymysql.cursors.Cursor)

    cursor = connection.cursor()

    query = "SELECT DISTINCT business.State FROM business"

    if flag == 'rating':
        query = "SELECT DISTINCT review.Rating FROM review"

    cursor.execute(query)

    categories = [state[0] for state in cursor]

    cursor.close()
    connection.close()

    return categories


all_cities = get_cities()


def write_document(cursor, cat, flag='state'):
    """ Given a yelp category, build out a text document
    which has all the reviews for that category """
    if flag == 'city':
        cities = all_cities[cat]
        n_encoding_errors = 0
        n_review = 0
        for city in cities:
            query = ("SELECT review.Content "
                     "FROM review INNER JOIN business "
                     "ON review.B_id = business.Business_id "
                     "WHERE business.State=%s AND business.City=%s")
            print(query)
            cursor.execute(query, (cat, city))

            with io.open(cat2doc(cat, flag, city), 'w', encoding='utf8') as f:
                for text, in cursor:
                    try:
                        f.write(text)
                        f.write("\n")
                    except UnicodeEncodeError:
                        n_encoding_errors += 1
                    n_review += 1
        return n_encoding_errors, n_review

    query = ("SELECT review.Content "
             "FROM review INNER JOIN business "
             "ON review.B_id = business.Business_id "
             "WHERE business.State=%s")
    if flag == 'rating':
        query = ("SELECT review.Content "
                 "FROM review "
                 "WHERE review.Rating=%s")

    cursor.execute(query, cat)

    n_encoding_errors = 0
    n_review = 0
    with io.open(cat2doc(cat, flag), 'w', encoding='utf8') as f:
        for text, in cursor:
            try:
                f.write(text)
                f.write("\n")
            except UnicodeEncodeError:
                n_encoding_errors += 1
            n_review += 1
    return n_encoding_errors, n_review


def document_text_iterator(categories, flag='state'):
    for filepath in cats2docs(categories,flag):
        with io.open(filepath, 'r', encoding='utf8') as f:
            yield f.read()


def document_iterator(categories, flag='state'):
    for filepath in cats2docs(categories, flag):
        yield filepath


def sql2txt(categories, flag='state'):
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 port=3306,
                                 password='suhuai2001',
                                 db='yelp',
                                 cursorclass=pymysql.cursors.Cursor)

    cursor = connection.cursor()

    if isinstance(categories, str):
        categories = (categories, )

    for cat in categories:
            n_errors, n_total = write_document(cursor, cat, flag)
            print("%s: %d errors, %d total" % (cat, n_errors, n_total))

    cursor.close()
    connection.close()


def create_all_documents(flag='state'):
    cats = all_categories(flag)
    # print("Creating %d documents" % len(cats))
    sql2txt(cats, flag)


def vectorize_sklearn(categories, flag='state'):
    # should I use the vocabulary from something like fasttext?
    vect = TfidfVectorizer(input='filename', preprocessor=preprocessor, tokenizer=stemmer,
                           vocabulary=None, token_pattern=None, stop_words=stop_words)
    X = vect.fit_transform(document_iterator(categories, flag))
    vocabulary = vect.get_feature_names_out()
    return X, categories, vocabulary


def save_pickle(matrix, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(matrix, outfile, pickle.HIGHEST_PROTOCOL)


def compute_and_save(flag='state'):
    categories = all_categories(flag)
    X, categories, vocabulary = vectorize_sklearn(categories, flag)
    save_pickle(X, 'tfidf/%s.mtx' % flag)
    np.savez_compressed('tfidf/%s-meta' % flag,
                        paths=cats2docs(categories, flag), vocabulary=vocabulary)


if __name__ == '__main__':
    # categories = ('Sushi Bars',
    #               'Bikes',
    #               'Dance Clubs')

    # create_all_documents(flag='rating')
    # create_all_documents(flag='state')
    # create_all_documents(flag='city')
    #compute_and_save('state')
    #compute_and_save('rating')
    compute_and_save('city')

'''
    categories = all_categories()
    X, categories, vocabulary = vectorize_sklearn(categories)

    save_pickle(X, 'tfidf/state.mtx')
    np.savez_compressed('tfidf/state-meta',
                        paths=cats2docs(categories), vocabulary=vocabulary)
'''