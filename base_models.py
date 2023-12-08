# Adapted from
# https://github.com/youralien/affinder-search/blob/4dd427c98c98533b0550c3d484fba55d75f39818/yelp_academic_etl_training.py
# Author youralien
import math
import os
import string
import re
import io
import numpy as np
import pickle
import pymysql.cursors
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from collections import Counter


'''
'''
connection = pymysql.connect(host='127.0.0.1',
                                user='root',
                                port=3306, # check out the port number
                                password='Jiayi-MySQL', # your password
                                db='yelp', # database name
                                cursorclass=pymysql.cursors.DictCursor)



def all_states():
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
    
    os.mkdir('tfidf/state')
    os.mkdir('tfidf/city')
    for state in states:
        path1 = 'tfidf/state' + '/' + state
        path2 = 'tfidf/city' + '/' + state
        os.mkdir(path1)
        os.mkdir(path2)
    '''

    cursor.close()
    return states



def get_cities():
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
            path = 'reviewtext/city' + '/' + state + '/' + city.replace('/', '-')
            os.mkdir(path)
        
        for city in cities[state]:
            path = 'tfidf/city' + '/' + state + '/' + city.replace('/', '-')
            os.mkdir(path)
        '''

    cursor.close()
    return cities


all_cities = get_cities()


def get_categories():
    categories_cities = {}
    categories_states = {}
    
    cursor = connection.cursor()
    states = all_states()
    query_state = ("SELECT DISTINCT Category_name "
                   "FROM category "
                   "INNER JOIN business "
                   "ON category.Business_id=business.Business_id "
                   "WHERE business.State=%s")

    for state in states:
        cursor.execute(query_state, state)
        categories_states[state] = [cat[0] for cat in cursor]

    query_city = ("SELECT DISTINCT Category_name "
                  "FROM category "
                  "INNER JOIN business "
                  "ON category.Business_id=business.Business_id "
                  "WHERE business.State=%s AND business.City=%s")

    for state in states:
        cur_state = {}
        for city in all_cities[state]:
            cursor.execute(query_city, (state, city))
            cur_state[city] = [cat[0] for cat in cursor]
        categories_cities[state] = cur_state

    cursor.close()
    return categories_states, categories_cities


categories_of_state, categories_of_city = get_categories()


def stemmer(text):
    ps = PorterStemmer()

    return ps.stem(text)


stop_words = [stemmer(i)[0] for i in ENGLISH_STOP_WORDS]


def preprocessor(text):
    pattern = r"[^A-Za-z\s'-]+"
    return stemmer(re.sub(pattern, "", text))


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
        return [cat2doc(state, cat, flag, city) for cat in categories]

    return [cat2doc(state, cat, flag) for cat in categories]


def write_document(cursor, state, cat, flag='state', city=None):
    """ Given a yelp category, build out a text document
    which has all the reviews for that category """
    if flag == 'city':
        n_encoding_errors = 0
        n_review = 0

        query = ("SELECT Content "
                 f"FROM {state.replace('N','Z')+city.translate(str.maketrans('', '', string.punctuation)).replace(' ','').replace('n','z').replace('N','Z')} "
                 "INNER JOIN category "
                 f"ON {state.replace('N','Z')+city.translate(str.maketrans('', '', string.punctuation)).replace(' ','').replace('n','z').replace('N','Z')}.B_id = category.Business_id "
                 "WHERE category.Category_name=%s")
        # print(query)
        cursor.execute(query, cat)

        with io.open(cat2doc(state, cat, flag, city), 'w', encoding='utf8') as f:
            for text, in cursor:
                try:
                    if text is not None:
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
             "WHERE business.State=%s AND category.Category_name=%s")

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


def document_text_iterator(state, categories, flag='state', city=None):
    for filepath in cats2docs(state, categories, flag, city):
        with io.open(filepath, 'r', encoding='utf8') as f:
            yield f.read()

def prepare_document_names(cat_of_setting, flag='state'):
    result = []
    if flag == 'state':
        states = all_states()
        for state in states:
            categories = cat_of_setting[state]
            for filepath in cats2docs(state, categories, flag):
                result.append(filepath)

    if flag == 'city':
        states = all_states()
        candidate_cities = non_empty_cities()
        for state in states:
            cites = candidate_cities[state]
            for city in cites:
                categories = cat_of_setting[state][city]
                for filepath in cats2docs(state, categories, flag, city):
                    result.append(filepath)

    return result


def sql2txt(states, flag='state'):

    cursor = connection.cursor()

    if isinstance(states, str):
        states = (states, )

    if flag == 'state':
        for state in states:
            categories = categories_of_state[state]
            for cat in categories:
                n_errors, n_total = write_document(cursor, state, cat, flag)
                print("%s: %d errors, %d total" % (state, n_errors, n_total))

    elif flag == 'city':
        for state in states:
            create_temp_state = (f"CREATE TEMPORARY TABLE {state.replace('N','Z')} AS "
                                 "SELECT review.B_id, review.Content, business.City "
                                 "FROM review "
                                 "INNER JOIN business "
                                 "ON review.B_id = business.Business_id "
                                 "WHERE business.State=%s")

            # print(create_temp_state, state)
            cursor.execute(create_temp_state, state)
            cities = all_cities[state]
            for city in cities:
                # .replace(' ', 'S').replace('-', 'W').replace(',', 'C').replace('.', 'P')
                create_temp_city = (f"CREATE TEMPORARY TABLE IF NOT EXISTS {state.replace('N','Z')+city.translate(str.maketrans('', '', string.punctuation)).replace(' ','').replace('n','z').replace('N','Z')} AS "
                                    "SELECT B_id, Content "
                                    f"FROM {state.replace('N','Z')} "
                                    "WHERE City=%s")
                print(create_temp_city)
                cursor.execute(create_temp_city, city)
                categories = categories_of_city[state][city]
                for cat in categories:
                    n_errors, n_total = write_document(cursor, state, cat, flag, city)
                    print("%s: %d errors, %d total" % (state, n_errors, n_total))

    else:
        print('unknown flag')

    cursor.close()


def create_all_documents(flag='state'):
    states = all_states()
    # print("Creating %d documents" % len(cats))
    sql2txt(states, flag)


def vectorize_sklearn(cats_of_setting, flag='state'):
    # should I use the vocabulary from something like fasttext?
    vect = TfidfVectorizer(input='filename', preprocessor=preprocessor, tokenizer=None,
                           vocabulary=None, token_pattern=r"[A-Za-z'-]+", stop_words=stop_words)
    document_names = prepare_document_names(cats_of_setting, flag)
    X = vect.fit_transform(document_names)
    vocabulary = vect.get_feature_names_out()
    return X, vocabulary


def save_pickle(matrix, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(matrix, outfile, pickle.HIGHEST_PROTOCOL)


def compute_and_save(threshold=1000, flag='state'):

    dir_path = 'tfidf/matrix_%s_%d/' % (flag, threshold)
    matrix_path = dir_path + '%s.mtx' % flag
    features_path = dir_path + '%s-features' % flag

    states_filtered, cities_filtered = get_categories_with_enough_reviews(threshold)
    cats_of_setting = states_filtered if flag == 'state' else cities_filtered
    X, vocabulary = vectorize_sklearn(cats_of_setting, flag)
    save_pickle(X, matrix_path)

    document_names = prepare_document_names(cats_of_setting, flag)
    np.savez_compressed(features_path, document_names=document_names, vocabulary=vocabulary)
    print(flag+' finished!')


def check_non_empty(state, city):
    my_path = "reviewtext/city/%s/%s" % (state, city.replace('/', '-'))
    for file in os.listdir(my_path):
        if file.endswith(".txt"):
            # print(os.path.join(my_path, file))
            text_path = os.path.join(my_path, file)
            with open(text_path) as f:
                lines = f.readlines()
                for line in lines:
                    if len(line) > 1:
                        return True
    return False


def non_empty_cities():
    new_dict = {}
    for state in all_cities.keys():
        new_dict[state] = [city for city in all_cities[state] if check_non_empty(state, city)]
    return new_dict


def get_review_distribution():
    states = all_states()

    size_list = []
    for state in states:
        my_path = "reviewtext/state/%s" % state
        for file in os.listdir(my_path):
            if file.endswith(".txt"):
                text_path = os.path.join(my_path, file)
                size_list.append(int(os.path.getsize(text_path)/6))

    print('median', np.median(size_list))
    print('upper', np.percentile(size_list, 75))
    print('lower', np.percentile(size_list, 25))

    log_size = [math.log10(e) for e in size_list]
    plt.boxplot(log_size)
    plt.show()
    '''
    
    log_size = [int(math.log10(e)) for e in size_list]
    C = Counter(log_size)
    plt.bar(C.keys(), C.values())
    plt.title('distribution of review size (state)')
    plt.xlabel("log10 word count")
    plt.ylabel("file count")
    plt.show()
    '''

    size_list = []
    cities_dict = non_empty_cities()
    for state in states:
        cities = cities_dict[state]
        for city in cities:
            my_path = "reviewtext/city/%s/%s" % (state, city.replace('/', '-'))
            for file in os.listdir(my_path):
                if file.endswith(".txt"):
                    text_path = os.path.join(my_path, file)
                    size_list.append((int(os.path.getsize(text_path)+1) / 6))
    print('median', np.median(size_list))
    print('upper', np.percentile(size_list, 75))
    print('lower', np.percentile(size_list, 25))
    log_size = [math.log10(e) for e in size_list]
    plt.boxplot(log_size)
    plt.show()


    '''
    C2 = Counter(log_size)
    plt.bar(C2.keys(), C2.values())
    plt.title('distribution of review size (city)')
    plt.xlabel("log10 word count")
    plt.ylabel("file count")
    plt.show()
    '''


def is_good_category(path,threshold=1000):
    return os.path.getsize(path) > threshold*6


def get_categories_with_enough_reviews(threshold=1000):
    states = all_states()
    cities_dict = non_empty_cities()
    good_categories_state = {}
    good_categories_city = {}
    for state in states:
        good_categories_state[state] = [cat for cat in categories_of_state[state] if is_good_category(cat2doc(state, cat),threshold)]

    for state in states:
        temp_dict = {}
        for city in cities_dict[state]:
            temp_dict[city] = [cat for cat in categories_of_city[state][city] if is_good_category(cat2doc(state, cat, 'city', city),threshold)]
        good_categories_city[state] = temp_dict

    '''
    size_list = []
    for state in states:
        for cat in good_categories_state[state]:
            size_list.append(int(os.path.getsize(cat2doc(state, cat)))/6)

    log_size = [int(math.log10(e)) for e in size_list]
    C = Counter(log_size)
    plt.bar(C.keys(), C.values())
    plt.title('distribution of review size (state)')
    plt.xlabel("log10 word count")
    plt.ylabel("file count")
    plt.show()
    '''
    return good_categories_state, good_categories_city


def save_states():
    states = all_states()
    np.savez_compressed('tfidf/state-meta', states=states)


def save_cities():
    candidate_cities = non_empty_cities()
    np.savez_compressed('tfidf/city-meta', all_cities=candidate_cities)


def save_categories():
    np.savez_compressed('tfidf/category-meta', categories_of_state=categories_of_state, categories_of_city=categories_of_city)


def save_good_categories(threshold=1000):
    file_name = f'tfidf/good-category-meta-%d' % threshold
    good_categories_state, good_categories_city = get_categories_with_enough_reviews(threshold)
    np.savez_compressed(file_name, good_categories_of_state=good_categories_state,
                        good_categories_of_city=good_categories_city)


if __name__ == '__main__':
    '''
    '''
    # categories = ('Sushi Bars',
    #               'Bikes',
    #               'Dance Clubs')
    create_all_documents(flag='state')
    # create_all_documents(flag='city')
    #compute_and_save('state')
    #compute_and_save('city')
    #save_categries()
    #save_states()
    #save_cities()
    #get_review_distribution()
    #get_categories_with_enough_reviews()
    #save_good_categories(3719)
    #save_good_categories()
    #compute_and_save(1000, 'state')
    #compute_and_save(1000, 'city')
    # compute_and_save(0, 'state')
    #print('Y')

    # check_non_empty('PA', 'West Norriton')
    #print(categories_of_city['IL']['Chicago'])
    connection.close()

