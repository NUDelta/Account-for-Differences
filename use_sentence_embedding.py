import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, util
import pickle
import re
import numpy as np

nlp = spacy.load("en_core_web_lg")
model = SentenceTransformer('all-mpnet-base-v2')

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


def load_good_categories():
    """
    Returns
    dict[state]: a list of good categories of that state
    dict[state][city]: a list of good categories of that city
    -------

    """
    loaded = np.load('tfidf/good-category-meta.npz', allow_pickle=True)
    return loaded['good_categories_of_state'], loaded['good_categories_of_city']


def cat2doc(state, cat, flag='state', city=None):
    """CA, Goleta, Sushi Bars -> reviewtext/city/CA/Goleta/SushiBars.txt
    From state,cat,(city) to Document filepath """

    path = "reviewtext/%s/%s" % (flag, state)

    if flag == 'city' and city is not None:
        # some city name somehow contains slashes for example Wayne/Radnor in PA.
        path = path + '/' + city.replace('/', '-')

    path = path + '/' + cat.replace('/', '-')
    return path + '.txt'


def retrieve_score_(filepath, words):
    texts = ''
    with open(filepath) as f:
        texts = f.read()

    #doc = nlp(texts)
    #sents = doc.sents
    #sentence_list = [sent.text for sent in sents]
    pattern = r'[\.?!]'
    sentence_list = re.split(pattern, texts)
    query_emb = model.encode(words)
    sentence_emb = model.encode(sentence_list)
    scores = util.cos_sim(query_emb, sentence_emb)
    return np.average(scores)


def retrieve_score(words, cat, state, flag='state', city=None):

    """
    Retrieve the tf-idf score of a word in relation to a category in a setting

    The preprocessor is necessary in order to match the precessing and tokenizing steps when calculating
    tf-idf score in base_models.py
    """
    filepath = cat2doc(state, cat, flag, city)
    return retrieve_score_(filepath, words)


if __name__ == '__main__':
    # print(retrieve_score('places to have fun', 'Beaches', 'FL')) 0.19
    # print(retrieve_score('places to have fun', 'Beaches', 'PA')) 0.13

    # print(retrieve_score('Where do families typically take their children to play in winter?', 'Beaches', 'FL')) 0.12
    # print(retrieve_score('Where do families typically take their children to play in winter?', 'Mobile home Dealers', 'FL')) 0.05


    # print(retrieve_score('Where do families typically take their children to play in winter?', 'Ski & Snowboard Shops', 'PA')) 0.086614065
    # print(retrieve_score('Where do families typically take their children to play in winter?', 'Ski Resorts','PA')) 0.12331266
    # print(retrieve_score('Where do families typically take their children to play in winter?', 'Skilled Nursing', 'PA')) 0.0588
    print(retrieve_score('Where do families typically take their children to play in winter?', 'Zoos', 'PA'))
    print('Y')
