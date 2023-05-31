# Account-for-Differences
This is the code base of the `DTR-HAT-Accounting for Differences` project. Last updated in Spring 2023.

To get started, run

`pip install -r requirements.txt`

---
Descriptions of each files
---

#### `use_tfidf`: system model based on tf-idf,with additional testing and analysis codes
- dependent on the files in the tfidf folder

#### `use_sentence_embedding`: system model based on document embedding, with additional codes that compute and save document embeddings
- run the following command before you run the python script: `pip install -U sentence-transformers`
- The scripts are complete, but you are not able to use retrieve_score function since we haven't computed the embeddings yet
- You can compute the embeddings and store them using the function `save_document_embeddings` but be cautious that it would many hours to compute them

#### `base model`: It basically does two things. 
- First, create documents by querying the local database we have constructed, where each document contains all the reviews from a category in a setting. 
- Second, compute tf-idf matrix that represents all the vectorized documents. Note that lots of functions take an argument called 'flag'. By default, flag is set to 'state', which means we are considering each state as a setting and we are doing state-wise comparison.

#### `load_database`: It first parses the JSON files from the Yelp open dataset, then connects to the local MySQL databse, create tables if they haven't been created, and inserts realtions into the tables.
- This script assumes that you have set up the local MySQL database

#### `find_reviews`: Given a word, a setting, and a context feature(yelp category), it searches for the relevant reviews and relevant sentences that contain the word.

#### `tf-idf`: This folder stores all the meta data needed to run the `use_tfidf` file, some file has a number suffix. 1000 means that only documents with more than 1000 words are considered

#### `reviewtext` This folder contains all the documents where each document is a setting-category pair. It's empty right now but the data exeeds the maximum capacity of the github repo. For each level of setting (state or city), it's 20 GB of data.

#### the `.dea` and `.DS_Store` are irrelevant and you can safely delete them

--- 

We also have [a Colab version prototype based on tf-idf](https://colab.research.google.com/drive/1bfuTUW5h2MkfhFTgGDx-dRzKCH49cD9j#scrollTo=yPjEmdqs3GlG). The tfidf directory that the Colab accesses has been moved from the `DTR` root folder to `DTR/Cubbies/Suhuai & Jiayi's Cubby`

This [doc](https://docs.google.com/document/d/1mkvDaruVGaCcO-Wqa1cK6arIJ1CW0_OJLU3eJGnWpCA/edit) contains the complete write-up of what we have done and also points to some future directions





