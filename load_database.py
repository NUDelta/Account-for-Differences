import json
import pymysql.cursors


if __name__ == '__main__':

    '''
    # create a new database
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 port=3306,
                                 password='suhuai2001')
    try:
        with connection.cursor() as cursor:
            cursor.execute('CREATE DATABASE yelp')

    finally:
        connection.close()

    '''

    # create the business table and review table
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 port=3306,
                                 password='suhuai2001',
                                 db='yelp',
                                 cursorclass=pymysql.cursors.DictCursor)

    try:
        with connection.cursor() as cursor:

            createBusinessTable = '''CREATE TABLE IF NOT EXISTS business(
                                                                Business_id CHAR(22),
                                                                Name TEXT, 
                                                                City VARCHAR(100), 
                                                                State CHAR(10),
                                                                PRIMARY KEY(Business_id))
                                                                '''
            cursor.execute(createBusinessTable)
            '''
            # insert business entries into business table
            with open('yelp_dataset/business.json') as f:
                for line in f:
                    current_object = json.loads(line)
                    business_id = current_object['business_id']
                    # escape single quotes
                    name = current_object['name'].replace("'", '"')
                    city = current_object['city'].replace("'", '"')
                    state = current_object['state'].replace("'", '"')

                    insertBusiness = f"INSERT INTO `business` (`Business_id`, `Name`, `City`, `State`) VALUES ('{business_id}','{name}','{city}','{state}')"
                    print(insertBusiness)
                    cursor.execute(insertBusiness)
                    connection.commit()
            '''

            #
            createReviewTable = '''CREATE TABLE IF NOT EXISTS review(
                                                                    Review_id CHAR(22),
                                                                    B_id CHAR(22),
                                                                    Rating INT(1), 
                                                                    Content TEXT,
                                                                    PRIMARY KEY(Review_id),
                                                                    FOREIGN KEY(b_id) REFERENCES business(business_id))
                                                                    '''
            cursor.execute(createReviewTable)

            with open('yelp_dataset/review.json') as f:
                for line in f:
                    current_object = json.loads(line)
                    review_id = current_object['review_id']
                    b_id = current_object['business_id']
                    rating = current_object['stars']
                    # escape single quotes
                    content = current_object['text'].replace("'", '"').replace('\n', ' ').replace('\\', '')

                    insertReview = f"INSERT INTO `review` (`Review_id`, `B_id`, `Rating`, `Content`) VALUES ('{review_id}','{b_id}','{rating}','{content}')"
                    print(insertReview)
                    cursor.execute(insertReview)
                    connection.commit()

    finally:
        connection.close()





# it contains business outside of US, eg. 'Liverpool, XMS'
#








