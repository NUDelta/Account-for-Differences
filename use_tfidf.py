import pickle
import re
import numpy as np
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import math
import matplotlib.pyplot as plt


def preprocessor(text):
    pattern = r"[^A-Za-z\s'-]+"
    return stemmer(re.sub(pattern, "", text))


def stemmer(text):
    ps = PorterStemmer()
    return ps.stem(text)


stop_words = [preprocessor(i) for i in ENGLISH_STOP_WORDS]


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


def load_good_categories(threshold=3719):
    """
    Returns
    dict[state]: a list of good categories of that state
    dict[state][city]: a list of good categories of that city
    -------

    """
    if threshold == 0:
        loaded = np.load('tfidf/category-meta.npz', allow_pickle=True)
        return loaded['categories_of_state'], loaded['categories_of_city']
    loaded = np.load('tfidf/good-category-meta-%d.npz' % threshold, allow_pickle=True)
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



def read_data(flag='state', threshold=3719):
    """
    given a state (and a city), load the related tf-idf matrix, filepaths and vocabulary
    where the filepaths are the row names and vocabulary contains the row names of the tf-idf matrix

    Output a tf-idf matrix, a dictionary that maps each filepath to the row index, and a dictionary that maps each
    word to the column index

    """

    dir_path = matrix_path = features_path = ''
    if threshold == 0:
        pass
    else:
        dir_path = 'tfidf/matrix_%s_%d/' % (flag, threshold)
        matrix_path = dir_path + '%s.mtx' % flag
        features_path = dir_path + '%s-features.npz' % flag

    

    with open(matrix_path, 'rb') as f:
        matrix = pickle.load(f)
        # matrix = matrix.todense()

    loaded = np.load(features_path, allow_pickle=True)

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
catToIndex_city, wordToIndex_city, matrix_city = read_data('city', 1000)


def retrieve_score_(filepath, words, catToIndex, wordToIndex, matrix):
    """
    retrieve the tf-idf score of a word in relation to a category when the matrix and mappings from names to indices are given
    """

    phrase_score = 0.0
    # words = preprocessor(words)
    word_list = re.findall(r"[A-Za-z'-]+", words)
    # print(word_list_old)
    # word_list = [preprocessor(i) for i in word_list_old]
    # print(word_list)

    # store each individual word score
    word_score_dict = {}

    if filepath not in catToIndex:
        # print("filepath doesn't exist: "+filepath)
        return "cat not found", word_score_dict

    x = catToIndex[filepath]

    for word in word_list:
        # stop_words shouldn't be considered
        if word in stop_words:
            # print(word, 'is a stop word')
            continue
        # wordToIndex is the vocabulary of all the documents.
        # If word is not in the vocabulary, it shouldn't exist in any of the documents
        if word not in wordToIndex:
            print(word, "doesn't exist")
            continue

        y = wordToIndex[word]

        # take log so that the product of multiplication would not underflow
        # add a small value to each of the word score so that log function is always defined
        # The small value can be considered as a penalty of missing the word too. The smaller the value, the harsher the penalty.
        word_score = math.log(matrix[x, y] + 10**(-6))
        # phrase_score += matrix[x, y]    # adding all
        phrase_score += word_score    # multiplying all
        word_score_dict[word] = word_score

    # print('phrase_score: ', phrase_score)
    # print('word_score_dict: ', word_score_dict)
    return phrase_score, word_score_dict


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


def getAll_desc(words, cats, state, flag='state', city=None):
    scores = []
    for cat in cats:
        # print('-'*20)
        # print('cat: ', cat)
        score, word_scores = retrieve_score(words, cat, state, flag, city)
        if score == 'cat not found':
            continue
        # print('score: ', score)
        scores.append((cat, score, word_scores))
    
    # initialize a dataframe
    df = pd.DataFrame(columns=['rank', 'cat', 'score'] + list(scores[0][2].keys()))
    for i in range(len(scores)):
        cat, score, word_scores = scores[i][0], scores[i][1], scores[i][2]
        df.loc[len(df)] = [0, cat, score] + list(word_scores.values())
    
    # rank the categories by their scores, and update rank
    df.sort_values(by=['score'], ascending=False, inplace=True)
    df['rank'] = range(1, len(df)+1)
    df.reset_index(drop=True, inplace=True)
    return df

    # scores.sort(key=lambda x: x[1], reverse=True)

    # cat_rank_score = {}
    # for i in range(len(scores)):
    #     cat, score = scores[i][0], scores[i][1]
    #     # if score == -1:
    #     #     continue
    #     cat_rank_score[cat] = (i+1, score)
    
    # return cat_rank_score


# data distribution
def get_data_distribution():
    # 1.1 get total number of states
    states = list(load_states())

    #1.2 get cats for each state
    # state_cats, city_cats = load_categories()
    state_cats, city_cats = load_good_categories()
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


def get_cats_city(state1, state2, city1, city2):
    state_cats, city_cats = load_good_categories(threshold=1000)
    city_cats = city_cats.item()
    city1_cats = city_cats[state1][city1]
    city2_cats = city_cats[state2][city2]
    
    # print(len(city1_cats), len(city2_cats))
    cats = list(set(city1_cats).union(set(city2_cats)))
    # print(len(cats))

    return cats


# make a bar plot for individual state
def generated_vs_expected(stateName, threshold):
    generated = stateName + '.csv'
    df = pd.read_csv(generated)
    # get the first 20 rows
    df_top20 = df.head(20)
    # print(df_top20)

    expected = stateName + '_expected.csv'
    # read the csv file with column names
    df_expected = pd.read_csv(expected)

    # make a histogram plot rank vs score, save it to a png file
    plt.bar(df['rank'], df['score'])
    # plt.xticks(range(0, 1200, 100))
    # plt.yticks(range(0, -140, -10))

    # # draw a horizontal line at y = -50
    # plt.axhline(y=-50, color='r', linestyle='--')
    # # draw a horizontal line at y = -60, -70, -80, -90, -100
    # plt.axhline(y=-60, color='r', linestyle='--')
    # plt.axhline(y=-70, color='r', linestyle='--')
    # plt.axhline(y=-80, color='r', linestyle='--')
    # plt.axhline(y=-90, color='r', linestyle='--')
    # plt.axhline(y=-100, color='r', linestyle='--')

    # highlight the expected categories with a vertical line
    for i in range(len(df_expected)):
        # if the rank <= threshold, green line
        # if the rank > threshold, red line, label it
        if df_expected.iloc[i]['rank'] <= threshold:
            plt.axvline(x=df_expected.iloc[i]['rank'], color='g', linestyle='--', label="in top "+str(threshold))
        else:
            plt.axvline(x=df_expected.iloc[i]['rank'], color='r', linestyle='--', label="out of top "+str(threshold))
            # put the label on the top of the line
            plt.text(df_expected.iloc[i]['rank'], int(df['score'].min()), df_expected.iloc[i]['cat'], rotation=45)

    plt.title(stateName)
    plt.xlabel('rank')
    plt.ylabel('score')
    # plt.legend(loc="upper left")
    # plt.legend()
    plt.savefig(stateName + '_bar.png')
    plt.close()
    

def expected_generator(expectedAns, cat_rank_score, state):
    # initialize a df with the columns of cat_rank_score.columns
    df_1 = pd.DataFrame(columns=cat_rank_score.columns)
    for ans in expectedAns:
        each = cat_rank_score[cat_rank_score['cat'].str.lower().str.contains(ans)]
        # concat each df to df_1
        df_1 = pd.concat([df_1, each], ignore_index=True)

    # result_1 = {x:[] for x in expectedAns}
    # cat_rank_score = cat_rank_score[[['rank', 'cat', 'score']]]
    # # loop this df, get each cat, rank, score
    # for row in cat_rank_score.iterrows():
    #     cat = row[1]['cat']
    #     rank = row[1]['rank']
    #     score = row[1]['score']
    #     for ans in expectedAns:
    #         if ans in cat.lower():
    #             result_1[ans].append([cat, rank, score])

    # for cat, rank_score in cat_rank_score.items():
    #     for ans in expectedAns:
    #         if ans in cat.lower():
    #             result_1[ans].append([cat, rank_score[0], rank_score[1]])

    # save result_1 to csv
    # df_1 = pd.DataFrame(columns=['expected ans', 'cat', 'rank', 'score'])
    # df_1 = pd.DataFrame(columns=['cat', 'rank', 'score'])
    # for ans, catRankScore in result_1.items():
    #     for each in catRankScore:
    #         df_1.loc[len(df_1.index)] = [each[0], each[1], each[2]]
    
    # print df_1 to check if exists any unexpected answers; if yes, enter cats want 
    # to drop in a list format, to drop them from df_1 and print df_1 again. 
    # If no, enter no to continue saving df_1 to csv
    print('Here is a list of expected answers for ' + state + ': \n', df_1)
    drop = input('Do you want to drop any answers? (y/n) ')
    while drop == 'y':
        drop_list = input('Enter the cats you want to drop (separated by comma without space): (ex. cat1,cat2,cat3) \n')
        drop_list = drop_list.split(',')
        df_1 = df_1[~df_1['cat'].isin(drop_list)]
        print('Here is a list of expected answers for ' + state + ': \n', df_1)
        drop = input('Do you want to drop any answers? (y/n) ')
    
    # order df_1 by rank
    df_1.sort_values(by='rank', inplace=True, ascending=True)
    df_1.reset_index(drop=True, inplace=True)
    
    df_1.to_csv(state + '_expected.csv', index=False)
    # print(result_1)
    print(state + ' -- number of expected answers: ', len(df_1))


# def resultTestcase(testCase, state1, expectedAns_1, state2, expectedAns_2, flag='state', city1=None, city2=None):
def resultTestcase(testCase, state1, state2, flag='state', city1=None, city2=None):
    print('-'*20)
    print(state1 + ' -- test case: ', testCase)
    df_1 = getAll_desc(testCase, cats, state1, flag, city1)
    # save cat_rank_score_1 to csv
    df_1.to_csv(state1 + '.csv', index=False)
    # print(len(df_1))

    print('-'*20)
    print(state2 + ' -- test case: ', testCase)
    df_2 = getAll_desc(testCase, cats, state2, flag, city2)
    # save cat_rank_score_2 to csv
    df_2.to_csv(state2 + '.csv', index=False)
    # print(len(df_2))

    # expected_generator(expectedAns_1, df_1, state1)
    # expected_generator(expectedAns_2, df_2, state2)


def zoomin_head_tail(state1, state2, df_state1, df_state2, df_state1_state2, threshold):
    df_state1_topk = df_state1.head(threshold)
    df_state2_topk = df_state2.head(threshold)

    df_head = df_state1_state2.head(threshold)
    df_tail = df_state1_state2.tail(threshold)

    cats = set(df_state1_topk['cat'].tolist() + df_state2_topk['cat'].tolist())
    print('len(cats): ', len(cats))
    # filter out rows that are not in cats
    df_head = df_head[df_head['cat'].isin(cats)]
    df_tail = df_tail[df_tail['cat'].isin(cats)]

    # concat df_head and df_tail
    df_head_tail = pd.concat([df_head, df_tail])
    df_head_tail.to_csv('head_tail.csv', index=False)

    # make a bar plot for rank vs difference
    plt.figure(figsize=(20, 10))
    plt.bar(df_head['rank'], df_head['difference'])
    # label each bar with cat
    for index, row in df_head.iterrows():
        plt.text(row['rank'], row['difference'], row['cat'], rotation=45)
    plt.xlabel('Diff. rank')
    plt.ylabel('Score of Difference')
    plt.title(state1 + ' top ' + str(threshold))
    plt.savefig(state1 + '_' + state2 + '_head.png')
    plt.close()

    # make a bar plot for rank vs difference
    plt.figure(figsize=(20, 10))
    plt.bar(df_tail['rank'], df_tail['difference'])
    # label each bar with cat
    for index, row in df_tail.iterrows():
        plt.text(row['rank'], row['difference'], row['cat'], rotation=45)
    plt.xlabel('Diff. rank')
    plt.ylabel('Score of Difference')
    plt.title(state2 + ' top ' + str(threshold))
    plt.savefig(state1 + '_' + state2 + '_tail.png')
    plt.close()


# '''
# - calculate the difference between state1 and state2 for each category and sort them by the difference
# - create a dataframe with columns: cat, state1_score, state2_score, difference, rank
# - save the dataframe to csv
# - make a bar plot for rank vs difference, save it to a png file
# '''
# def compare_two_states(state1, state2, threshold):
#     state1_csv = state1 + '.csv'
#     state2_csv = state2 + '.csv'
#     state1_expected_csv = state1 + '_expected.csv'
#     state2_expected_csv = state2 + '_expected.csv'

#     # read csv files
#     df_state1 = pd.read_csv(state1_csv)
#     df_state2 = pd.read_csv(state2_csv)
#     df_state1_expected = pd.read_csv(state1_expected_csv)
#     df_state2_expected = pd.read_csv(state2_expected_csv)

#     # initialize a dataframe with columns: rank, cat, state1_score, state2_score, difference
#     s1_cols = [state1 + '_' + x for x in df_state1.columns[3:]]
#     s2_cols = [state2 + '_' + x for x in df_state2.columns[3:]]
#     df_cols = ['rank', 'cat', state1+'_score', state2+'_score', 'difference'] + s1_cols + s2_cols
#     df = pd.DataFrame(columns=df_cols)

#     # get union of categories
#     cats = list(set(df_state1['cat'].tolist() + df_state2['cat'].tolist()))
#     print('number of categories: ', len(cats))
#     # get state1_score and state2_score for each category
#     for cat in cats:
#         state1_score = state2_score = diff = 0
#         ws_1 = ws_2 = [0] * len(df_state1.iloc[0][3:].tolist())
#         if cat in set(df_state1['cat'].tolist()):
#             state1_score = df_state1.loc[df_state1['cat'] == cat, 'score'].iloc[0]
#             ws_1 = df_state1.loc[df_state1['cat'] == cat].iloc[0][3:].tolist()
#         else:
#             state1_score = df_state1['score'].min() - 1

#         if cat in set(df_state2['cat'].tolist()):
#             state2_score = df_state2.loc[df_state2['cat'] == cat, 'score'].iloc[0]
#             ws_2 = df_state2.loc[df_state2['cat'] == cat].iloc[0][3:].tolist()
#         else:
#             state2_score = df_state2['score'].min() - 1
        
#         diff = state1_score - state2_score
#         df.loc[len(df.index)] = [0, cat, state1_score, state2_score, diff] + ws_1 + ws_2
    
#     # sort df_all by difference and update rank
#     df.sort_values(by=['difference'], inplace=True, ascending=False)
#     df.reset_index(drop=True, inplace=True)
#     for index, row in df.iterrows():
#         df.loc[index, 'rank'] = index + 1

#     # save df, df_NA, and df_all to csv
#     df.to_csv(state1 + '_' + state2 + '.csv', index=False)

#     # make a bar plot for rank vs difference
#     plt.figure(figsize=(20, 10))
#     plt.bar(df['rank'], df['difference'])
#     # plt.xticks(range(0, 1250, 100))
#     # plt.yticks(range(65, -65, -5))

#     # generate df_state1_state2_expected from df
#     df_state1_state2_expected = pd.DataFrame(columns=df_cols)
#     # get rows of df whose cat is in df_state1_expected or df_state2_expected
#     for index, row in df.iterrows():
#         cat = row['cat']
#         if cat in set(df_state1_expected['cat'].tolist() + df_state2_expected['cat'].tolist()):
#             df_state1_state2_expected.loc[len(df_state1_state2_expected.index)] = row
#     # order df_state1_state2_expected by rank
#     df_state1_state2_expected.sort_values(by=['rank'], inplace=True, ascending=True)
#     df_state1_state2_expected.reset_index(drop=True, inplace=True)
#     # save df_state1_state2_expected to csv
#     df_state1_state2_expected.to_csv(state1 + '_' + state2 + '_expected.csv', index=False)

#     count = 0
#     # highlight the expected cats 1 with a cyan vertical line
#     for cat in df_state1_expected['cat'].tolist():
#         rank = df.loc[df['cat'] == cat, 'rank'].iloc[0]
#         plt.axvline(x=rank, color='c', linestyle='--', label='Expected Cats for ' + state1)
#         if count % 2 == 0:        
#             plt.text(rank, int(df['difference'].min()), cat, rotation=45)
#         else:
#             plt.text(rank, int(df['difference'].max()), cat, rotation=45)
#         count += 1
#     # highlight the expected cats 2 with a yellow vertical line
#     for cat in df_state2_expected['cat'].tolist():
#         rank = df.loc[df['cat'] == cat, 'rank'].iloc[0]
#         plt.axvline(x=rank, color='y', linestyle='--', label='Expected Cats for ' + state2)
#         if count % 2 == 0:        
#             plt.text(rank, int(df['difference'].min()), cat, rotation=45)
#         else:
#             plt.text(rank, int(df['difference'].max()), cat, rotation=45)
#         count += 1
        
#     plt.xlabel('Diff. rank')
#     plt.ylabel('Score of Difference')
#     plt.title(state1 + ' vs. ' + state2)
#     # plt.legend()
#     plt.savefig(state1 + '_' + state2 + '.png')
#     plt.close()

#     zoomin_head_tail(state1, state2, df_state1, df_state2, df, threshold)


#     # # get expected cats' rank
#     # expected_diff = pd.DataFrame(columns=['rank', 'cat', state1+'_score', state2+'_score', 'difference'])
#     # for cat in expectedCats_1 + expectedCats_2:
#     #     if cat not in df['cat'].tolist():
#     #         print(cat)
#     #         row = df_NA.loc[df_NA['cat'] == cat].iloc[0]
#     #         expected_diff.loc[len(expected_diff)] = list(row)
#     #         continue

#     #     # get row of this cat in df
#     #     row = df.loc[df['cat'] == cat].iloc[0]
#     #     # store into expected_diff use .loc
#     #     expected_diff.loc[len(expected_diff)] = list(row)

#     # # save expected_diff to csv
#     # expected_diff.to_csv(state1 + '_' + state2 + '_expected.csv', index=False)



'''
- calculate the difference between state1 and state2 for each category and sort them by the difference
- create a dataframe with columns: cat, state1_score, state2_score, difference, rank
- save the dataframe to csv
- make a bar plot for rank vs difference, save it to a png file
'''
def compare_two_states(state1, state2, param_setting1_topX, param_setting1_2_topY, intersection_topZ):
    state1_csv = state1 + '.csv'
    state2_csv = state2 + '.csv'

    # read csv files
    df_state1 = pd.read_csv(state1_csv)
    df_state2 = pd.read_csv(state2_csv)

    # initialize a dataframe with columns: rank, cat, state1_score, state2_score, difference
    s1_cols = [state1 + '_' + x for x in df_state1.columns[3:]]
    s2_cols = [state2 + '_' + x for x in df_state2.columns[3:]]
    df_cols = ['rank', 'cat', state1+'_score', state2+'_score', 'difference'] + s1_cols + s2_cols
    df = pd.DataFrame(columns=df_cols)

    # get union of categories
    cats = list(set(df_state1['cat'].tolist() + df_state2['cat'].tolist()))
    print('number of categories: ', len(cats))
    # get state1_score and state2_score for each category
    for cat in cats:
        state1_score = state2_score = diff = 0
        ws_1 = ws_2 = [0] * len(df_state1.iloc[0][3:].tolist())
        if cat in set(df_state1['cat'].tolist()):
            state1_score = df_state1.loc[df_state1['cat'] == cat, 'score'].iloc[0]
            ws_1 = df_state1.loc[df_state1['cat'] == cat].iloc[0][3:].tolist()
        else:
            state1_score = df_state1['score'].min() - 1

        if cat in set(df_state2['cat'].tolist()):
            state2_score = df_state2.loc[df_state2['cat'] == cat, 'score'].iloc[0]
            ws_2 = df_state2.loc[df_state2['cat'] == cat].iloc[0][3:].tolist()
        else:
            state2_score = df_state2['score'].min() - 1
        
        diff = state1_score - state2_score
        df.loc[len(df.index)] = [0, cat, state1_score, state2_score, diff] + ws_1 + ws_2
    
    # get rows which state1_score != 0 and state2_score == 0, store them to df_state1_prevalence
    df_state1_pre = df.loc[(df[state1+'_score'] != df[state1+'_score'].min()) & (df[state2+'_score'] == df[state2+'_score'].min())]
    # sort by state1_score and update rank
    df_state1_pre.sort_values(by=[state1+'_score'], inplace=True, ascending=False)
    df_state1_pre.reset_index(drop=True, inplace=True)
    for index, row in df_state1_pre.iterrows():
        df_state1_pre.loc[index, 'rank'] = index + 1
    df_state1_pre.to_csv(state1 + '_prevalence.csv', index=False)

    # get rows which state1_score == 0 and state2_score != 0, store them to df_state2_prevalence
    df_state2_pre = df.loc[(df[state1+'_score'] == df[state1+'_score'].min()) & (df[state2+'_score'] != df[state2+'_score'].min())]
    # sort by state2_score and update rank
    df_state2_pre.sort_values(by=[state2+'_score'], inplace=True, ascending=False)
    df_state2_pre.reset_index(drop=True, inplace=True)
    for index, row in df_state2_pre.iterrows():
        df_state2_pre.loc[index, 'rank'] = index + 1
    df_state2_pre.to_csv(state2 + '_prevalence.csv', index=False)

    # remove rows which in df_state1_pre and df_state2_pre from df
    df = df.loc[(df[state1+'_score'] != df[state1+'_score'].min()) & (df[state2+'_score'] != df[state2+'_score'].min())]
    # sort df_all by difference and update rank
    df.sort_values(by=['difference'], inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    for index, row in df.iterrows():
        df.loc[index, 'rank'] = index + 1

    # save df, df_NA, and df_all to csv
    df.to_csv(state1 + '_' + state2 + '.csv', index=False)
    
    # get cats of df which is in the top threshold in df_state1
    # df_state1_top = df_state1.loc[df_state1['rank'] <= param_setting1_topX]
    # df_state1_relevant stores rows which score != min score
    df_state1_relevant = df_state1.loc[df_state1['score'] != df_state1['score'].min()]
    # df_state1_top = df_state1_relevant.iloc[:int(param_setting1_topX * len(df_state1_relevant))]
    df_state1_top_cats = df_state1_relevant['cat'].tolist()

    # df_head_Y = df.loc[df['rank'] <= param_setting1_2_topY]
    # df_state1_has_diff = df.loc[df['difference'] > 0]
    # df_head_Y = df_state1_has_diff.iloc[:int(param_setting1_2_topY * len(df_state1_has_diff))]
    df_state1_diff = df.loc[df['cat'].isin(df_state1_top_cats)]

    # order df_state1_diff by diff score and update rank
    df_state1_diff.sort_values(by=['difference'], inplace=True, ascending=False)
    df_state1_diff.reset_index(drop=True, inplace=True)

    # update the rank
    for index, row in df_state1_diff.iterrows():
        df_state1_diff.loc[index, 'rank'] = index + 1
    # pick top 20 
    # intersection_num = int(intersection_topZ * len(df_state1_diff))
    # if intersection_num < 1:
    #     intersection_num = 1
    # df_state1_diff_topZ = df_state1_diff.iloc[:intersection_num]

    # save df_state1_diff to csv
    # df_state1_diff.to_csv(state1 + '_' + str(param_setting1_topX) + '_' + str(param_setting1_2_topY) + '_diff.csv', index=False)
    df_state1_diff.to_csv(state1 + '_diff.csv', index=False)
    # df_state1_diff_topZ.to_csv(state1 + '_' + str(param_setting1_topX) + '_' + str(param_setting1_2_topY) + '_diff_' + str(intersection_topZ) + '.csv', index=False)

    '''
    '''
    # get cats of df which is in the top threshold in df_state2
    # df_state2_top = df_state2.loc[df_state2['rank'] <= param_setting1_topX]
    # df_state2_relevant stores rows which score != min score
    df_state2_relevant = df_state2.loc[df_state2['score'] != df_state2['score'].min()]
    # df_state2_top = df_state2_relevant.iloc[:int(param_setting1_topX * len(df_state2_relevant))]
    df_state2_top_cats = df_state2_relevant['cat'].tolist()

    # df_state2_has_diff = df.loc[df['difference'] < 0]
    # df_tail_Y = df_state2_has_diff.iloc[(-1) * int(param_setting1_2_topY * len(df_state2_has_diff)):]
    df_state2_diff = df.loc[df['cat'].isin(df_state2_top_cats)]

    # order df_state2_diff by diff score and update rank
    df_state2_diff.sort_values(by=['difference'], inplace=True, ascending=True)
    df_state2_diff.reset_index(drop=True, inplace=True)
    # update the rank
    for index, row in df_state2_diff.iterrows():
        df_state2_diff.loc[index, 'rank'] = index + 1

    # pick top 20
    # intersection_num = int(intersection_topZ * len(df_state2_diff))
    # if intersection_num < 1:
    #     intersection_num = 1
    # df_state2_diff_topZ = df_state2_diff.iloc[:intersection_num]

    # save df_state2_diff to csv
    df_state2_diff.to_csv(state2 + '_diff.csv', index=False)
    # df_state2_diff.to_csv(state2 + '_' + str(param_setting1_topX) + '_' + str(param_setting1_2_topY) + '_diff.csv', index=False)
    # df_state2_diff_topZ.to_csv(state2 + '_' + str(param_setting1_topX) + '_' + str(param_setting1_2_topY) + '_diff_' + str(intersection_topZ) + '.csv', index=False)


def compare_expected(path, state1, state2):
    state1_csv = path + state1 + '_expected.csv'
    state2_csv = path + state2 + '_expected.csv'

    # read csv files
    df_state1 = pd.read_csv(state1_csv)
    df_state2 = pd.read_csv(state2_csv)

    cats = set(df_state1['cat'].tolist() + df_state2['cat'].tolist())

    state1_state2_csv = path + state1 + '_' + state2 + '.csv'
    df_state1_state2 = pd.read_csv(state1_state2_csv)

    # initialize a dataframe to store df_state1_state2.columns
    df_expected = pd.DataFrame(columns=df_state1_state2.columns)
    # for each cat in cats
    for cat in cats:
        # if cat is not in df_state1_state2, only store cat's name
        if cat not in df_state1_state2['cat'].tolist():
            df_expected.loc[len(df_expected)] = [0, cat, 0, 0, 0]
            continue
        # get row of this cat in df_state1_state2
        row = df_state1_state2.loc[df_state1_state2['cat'] == cat].iloc[0]
        # store into df_expected use .loc
        df_expected.loc[len(df_expected)] = list(row)
    # save df_expected to csv
    df_expected.to_csv(path + state1 + '_' + state2 + '_expected2.csv', index=False)





if __name__ == '__main__':
    # # path = 'results/filter3719+good3719/FL_PA/'
    # # compare_two_states_100(path, 'FL', 'PA')
    # # compare_expected(path, 'FL', 'PA')
    # retrieve_score("Where do families typically take their children to play in winter?",'Indoor Playcentre', 'FL')
    # print('-'*100)

    '''
    1. get data distribution
    '''
    cats = get_data_distribution()
    # # cats = get_cats_city('LA', 'CA', 'New Orleans', 'Santa Barbara')
    
    '''
    2. generating results for test case
    '''
    '''
    2.1. test case 1
    '''
    # testCase = 'place or activity people will go or do for fun in winter'
    # testCase = 'place to have fun in winter'
    # testCase = 'place to go in winter'

    testCase = 'where to sunbathe'
    # testCase = 'place to sunbathe'
    setting1 = 'FL'
    setting2 = 'PA'
    param_setting1_topX = 0.2
    param_setting1_2_topY = 1   # already more relevant to FL than PA
    intersection_topZ = 1

    # # expected answers for FL
    # expectedAns_FL = [
    #     'beach',
    #     'park',
    #     'aquarium',
    #     'zoo'
    # ]

    # # expected answers for PA
    # expectedAns_PA = [
    #     'ski',
    #     'skat',
    #     'bik',
    #     'museum',
    #     'park'
    # ]
    # resultTestcase(testCase, 'FL', expectedAns_FL, 'PA', expectedAns_PA)
    resultTestcase(testCase, setting1, setting2)

    # '''
    # 2.2. test case 2
    # '''
    # # testCase = 'Affordable food for a party'
    # # expected answers for New Orleans
    # # expectedAns_NO = [
    # #     'Cajun-Creole',
    # #     'Chicken Shop',
    # #     'Chicken Wings',
    # #     'Soul food',
    # #     'Pizza',
    # #     'Fast Food',
    # #     'American'
    # # ]
    # # # expected answers for SB
    # # expectedAns_SB = [
    # #     'Tacos',
    # #     'Mexican',
    # #     'Caribbean',
    # #     'Pizza',
    # #     'Fast Food',
    # #     'American'
    # # ]
    # # expectedAns_NO = [
    # #     'cajun',
    # #     'chicken',
    # #     'soul food',
    # #     'pizza',
    # #     'fast food',
    # #     'american'
    # # ]
    # # # expected answers for SB
    # # expectedAns_SB = [
    # #     'tacos',
    # #     'mexican',
    # #     'caribbean',
    # #     'pizza',
    # #     'fast food',
    # #     'american'
    # # ]

    # # resultTestcase(testCase, 'LA', expectedAns_NO, 'CA', expectedAns_SB, flag='city', city1='New Orleans', city2='Santa Barbara')


    # '''
    # 3. make plots for each state in the test case
    # '''
    # # expectedCats_FL = ['Zoos', 'Aquariums', 'Aquarium Services', 'Beach Equipment Rentals', 'Beaches']
    # # rank threshold = 100
    # generated_vs_expected('FL', 100)

    # # expectedCats_PA = ['Trampoline Parks', 'Children"s Museums', 'Ski & Snowboard Shops', 'Ski Resorts', 'Water Parks', 'Skating Rinks']
    # generated_vs_expected('PA', 100)

    # # expectedCats_No = ['Cajun/Creole', 'Chicken Shop', 'Chicken Wings', 'Soul Food']
    # # expectedCats_SB = ['Tacos', 'Mexican', 'Caribbean']
    # # generated_vs_expected('LA', expectedCats_No)
    # # generated_vs_expected('CA', expectedCats_SB)


    '''
    4. compare two states in the test case and make plots
    '''
    # threshold = 100: top 100 in setting 1 or 2, and top 100 in diff ranking
    compare_two_states(setting1, setting2, param_setting1_topX, param_setting1_2_topY, intersection_topZ)
    # compare_two_states('FL', 'PA', 10)

    # compare_two_states('LA', expectedCats_No, 'CA', expectedCats_SB)


    


    # for testCase in testCases:
    #     print('-'*20)
    #     print('test case: ', testCase)
    #     result_FL = get_top_k(testCase, 10, cats, 'FL')
    #     result_PA = []
    #     for cat, score in result_FL:
    #         result_PA.append((cat, retrieve_score(testCase, cat, 'PA')))
    #     df = pd.DataFrame(columns=['FL', 'PA', 'Diff'])
    #     for i in range(len(result_FL)):
    #         df.loc[len(df.index)] = [result_FL[i], result_PA[i], result_FL[i][1] - result_PA[i][1]]
    #     print(df)
        
    # c, w, m = read_data('AZ')
    # print(c.keys())

    #print(retrieve_score('fun', 'Restaurants', 'AZ'))

    # print(retrieve_score(words, 'Restaurants', 'AZ'))
    # print(retrieve_score(words, 'Restaurants', 'CA', 'city', 'Goleta'))
    # input('press enter to continue')


    # testCases = [
    #     'Food that make people fat',
    #     'People get fat by',
    #     'Food that are healthy for people',
    #     'Food that cheers you up',

    #     'Places to exhaust children',
    #     'Places for a private conversation',
    #     'Places to have a break up',
        
    #     'Fun things to do',
    #     'Thing that make you homesick',
    #     'Creepy experience',
    #     'The moments when you feel safe',
    #     'Wholesome weekend',
    #     'Dress like a gentleman'
    # ]

# testCase = 'Where do families typically take their children to play in winter?'
# state = 'FL'
# cats = ['Drive-In Theater', 'Virtual Reality Centers', 'Shared Office Spaces', 
#         'Real Estate Law', 'Race Tracks', 'Tabletop Games', 'Pool & Billiards', 
#         'Occupational Therapy', 'Themed Cafes', 'Pop-up Shops']
# retrieve_score(testCase, cats, state, flag='state', city=None)



    
    # FL_csv = 'results/filter3719+good3719/FL_PA_new2/FL.csv'
    # PA_csv = 'results/filter3719+good3719/FL_PA_new2/PA.csv'
    # all_csv = 'results/filter3719+good3719/FL_PA_new2/FL_PA.csv'
    # # read csv to df
    # df_FL = pd.read_csv(FL_csv)
    # df_PA = pd.read_csv(PA_csv)
    # df_all = pd.read_csv(all_csv)

    # # get df_all rows which in the top 100 cats of df_FL
    # df_FL_top100 = df_FL.head(100)
    # df_FL_top100_cats = df_FL_top100['cat'].tolist()
    # df_all_top100 = df_all[df_all['cat'].isin(df_FL_top100_cats)]
    # # rank df_all_top100 by diff
    # df_all_top100 = df_all_top100.sort_values(by=['difference'], ascending=False)
    # # save to FL_diff_100.csv
    # df_all_top100.to_csv('FL_diff_100.csv', index=False)

    # # get df_all rows which in the top 100 cats of df_PA
    # df_PA_top100 = df_PA.head(100)
    # df_PA_top100_cats = df_PA_top100['cat'].tolist()
    # df_all_tail100 = df_all[df_all['cat'].isin(df_PA_top100_cats)]
    # # rank df_all_tail100 by diff
    # df_all_tail100 = df_all_tail100.sort_values(by=['difference'], ascending=True)
    # # save to PA_diff_100.csv
    # df_all_tail100.to_csv('PA_diff_100.csv', index=False)

    # # df_FL count values for typically and winter columns
    # df_FL_count = df_FL[['typically', 'winter']].apply(pd.Series.value_counts)
    # print('-'*20)
    # print('FL')
    # print(df_FL_count)

    # # df_PA count values for typically and winter columns
    # df_PA_count = df_PA[['typically', 'winter']].apply(pd.Series.value_counts)
    # print('-'*20)
    # print('PA')
    # print(df_PA_count)
