import re
import io
import string
import pymysql.cursors


def get_review_sentence(word, cat, state, flag='state', city=None):
    path = cat2doc(state, cat, flag, city)
    pattern = r'[\.?!]'
    first_cap = word.capitalize()
    all_cap = word.upper()
    results = []

    with io.open(path)as f:
        text = f.read()
        sentences = re.split(pattern, text)
        for sentence in sentences:
            if word in sentence or first_cap in sentence or all_cap in sentence:
                results.append(sentence)

    return results


def get_review_exact_match(word, cat, state, flag='state', city=None):
    first_cap = word.capitalize()
    all_cap = word.upper()
    results = []

    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 port=3306,
                                 password='suhuai2001',
                                 db='yelp',
                                 cursorclass=pymysql.cursors.Cursor)

    cursor = connection.cursor()

    if flag == 'state':

        query = ("SELECT review.Content "
                 "FROM review "
                 "INNER JOIN (business "
                 "INNER JOIN category "
                 "ON business.Business_id = category.Business_id) "
                 "ON review.B_id = business.Business_id "
                 "WHERE business.State=%s AND category.Category_name=%s")

        cursor.execute(query, (state, cat))

    if flag == 'city':

        create_temp_state = (f"CREATE TEMPORARY TABLE {state.replace('N', 'Z')} AS "
                             "SELECT review.B_id, review.Content, business.City "
                             "FROM review "
                             "INNER JOIN business "
                             "ON review.B_id = business.Business_id "
                             "WHERE business.State=%s")

        cursor.execute(create_temp_state, state)

        create_temp_city = (
            f"CREATE TEMPORARY TABLE IF NOT EXISTS {state.replace('N', 'Z') + city.translate(str.maketrans('', '', string.punctuation)).replace(' ', '').replace('n', 'z').replace('N', 'Z')} AS "
            "SELECT B_id, Content "
            f"FROM {state.replace('N', 'Z')} "
            "WHERE City=%s")

        cursor.execute(create_temp_city, city)

        query = ("SELECT Content "
                 f"FROM {state.replace('N', 'Z') + city.translate(str.maketrans('', '', string.punctuation)).replace(' ', '').replace('n', 'z').replace('N', 'Z')} "
                 "INNER JOIN category "
                 f"ON {state.replace('N', 'Z') + city.translate(str.maketrans('', '', string.punctuation)).replace(' ', '').replace('n', 'z').replace('N', 'Z')}.B_id = category.Business_id "
                 "WHERE category.Category_name=%s")

        cursor.execute(query, cat)

    for text, in cursor:
        if word in text or first_cap in text or all_cap in text:
            results.append(text)

    return results


def cat2doc(state, cat, flag='state', city=None):
    """Sushi Bars -> Sushi Bars.txt
    From Category to Document filepath """

    path = "reviewtext/%s/%s" % (flag, state)

    if flag == 'city' and city is not None:
        # some city name somehow contains slashes for example Wayne/Radnor in PA.
        path = path + '/' + city.replace('/', '-')

    path = path + '/' + cat.replace('/', '-')
    return path + '.txt'

if __name__ == '__main__':

    '''
    ls = get_review_sentence('fun', 'Paint & Sip', 'CA')
    for l in ls:
        print(l)
    '''

    rs = get_review_exact_match('private', 'Speakeasies', 'FL')
    for r in rs:
        print(r)

    print('N')
