import pickle
import re
import numpy as np
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def preprocessor(text):
    pattern = r"[^A-Za-z\s'-]+"
    return stemmer(re.sub(pattern, "", text))


def stemmer(text):
    ps = PorterStemmer()
    return ps.stem(text)


stop_words = [stemmer(i)[0] for i in ENGLISH_STOP_WORDS]


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
        # print("filepath doesn't exist: "+filepath)
        return -1

    x = catToIndex[filepath]

    for word in word_list:
        if word in stop_words or word not in wordToIndex:
            # print("word doesn't exist: "+word)
            # return -2
            continue

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


def get_top_k(words, k, cats, state, flag='state', city=None):
    '''
    input a string of words and return the top k categories that are most related to the words
    '''
    if flag == 'state':
        catToIndex, wordToIndex, matrix = catToIndex_state, wordToIndex_state, matrix_state
    if flag == 'city':
        catToIndex, wordToIndex, matrix = catToIndex_city, wordToIndex_city, matrix_city
    
    scores = []
    for cat in cats:
        score = retrieve_score(words, cat, state)
        scores.append((cat, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]

    # # store the top k categories with their scores to a csv file
    # with open('top_k.csv', 'w') as f:
    #     for i in range(k):
    #         if i < len(scores):
    #             f.write('%s,%s\n' % (scores[i][0], scores[i][1]))



# data distribution
def get_data_distribution():
    # 1.1 get total number of states
    states = list(load_states())

    #1.2 get cats for each state
    state_cats, city_cats = load_categories()
    state_cats = state_cats.item()

    # 1.3 create a dataframe has state, state_cat, number of cats in each state
    df = pd.DataFrame(columns=['state', 'state_cat', 'num_cats'])
    for state, cats in state_cats.items():
        df.loc[len(df.index)] = [state, cats, len(cats)]
    
    # 1.4 save the dataframe to csv
    df.to_csv('state_cat.csv', index=False)

    # 1.5 
    print('-'*20)
    # print the number of states
    print('number of states: ', len(states))
    # print the state with the most categories, and print how many categories it has
    # print('state with the most categories: ', df[df['num_cats'] == df['num_cats'].max()]['state'].values[0])
    # print('number of categories: ', df[df['num_cats'] == df['num_cats'].max()]['num_cats'].values[0])
    print('state with the most categories: ', df.sort_values(by='num_cats', ascending=False).head(5))

    # print the state with the least categories, and print how many categories it has
    print('state with the least categories: ', df[df['num_cats'] == df['num_cats'].min()]['state'].values[0])
    print('number of categories: ', df[df['num_cats'] == df['num_cats'].min()]['num_cats'].values[0])

    # 2.1 get total number of cats 
    cats = set()
    for state, state_cat in state_cats.items():
        cats = cats.union(state_cat)
    
    # 2.2 get states for each cat
    cat_states = {}
    for cat in cats:
        cat_states[cat] = []
        for state, state_cat in state_cats.items():
            if cat in state_cat:
                cat_states[cat].append(state)

    # 2.3 create a dataframe has cat, cat_state, number of states for each cat
    df = pd.DataFrame(columns=['cat', 'cat_state', 'num_states'])
    for cat, states in cat_states.items():
        df.loc[len(df.index)] = [cat, states, len(states)]

    # 2.4 save the dataframe to csv
    df.to_csv('cat_state.csv', index=False)

    # 2.5
    print('-'*20)
    # print the number of cats
    print('number of cats: ', len(cats))
    # print the cat with the most states, and print how many states it has
    # print('cat with the most states: ', df[df['num_states'] == df['num_states'].max()]['cat'].values[0])
    # print('number of states: ', df[df['num_states'] == df['num_states'].max()]['num_states'].values[0])
    print('cat with the most states: ', df.sort_values(by='num_states', ascending=False).head(5))

    # print the cat with the least states, and print how many states it has
    print('cat with the least states: ', df[df['num_states'] == df['num_states'].min()]['cat'].values[0])
    print('number of states: ', df[df['num_states'] == df['num_states'].min()]['num_states'].values[0])

    return cats


if __name__ == '__main__':
    cats = get_data_distribution()

    '''
    1. State: FL, PA (since they have the most categories)
    '''
    testCases = [
        'Food that make people fat',
        'People get fat by',
        'Food that are healthy for people',
        'Food that cheers you up',

        'Places to exhaust children',
        'Places for a private conversation',
        'Places to have a break up',
        
        'Fun things to do',
        'Thing that make you homesick',
        'Creepy experience',
        'The moments when you feel safe',
        'Wholesome weekend',
        'Dress like a gentleman'
    ]
    
    for testCase in testCases:
        print('-'*20)
        print('test case: ', testCase)
        result_FL = get_top_k(testCase, 5, cats, 'FL')
        result_PA = []
        for cat, score in result_FL:
            result_PA.append((cat, retrieve_score(testCase, cat, 'PA')))
        df = pd.DataFrame(columns=['FL', 'PA', 'Diff'])
        for i in range(len(result_FL)):
            df.loc[len(df.index)] = [result_FL[i], result_PA[i], result_FL[i][1] - result_PA[i][1]]
        print(df)
        


    # c, w, m = read_data('AZ')
    # print(c.keys())

    #print(retrieve_score('fun', 'Restaurants', 'AZ'))

    # print(retrieve_score(words, 'Restaurants', 'AZ'))
    # print(retrieve_score(words, 'Restaurants', 'CA', 'city', 'Goleta'))
    # input('press enter to continue')

