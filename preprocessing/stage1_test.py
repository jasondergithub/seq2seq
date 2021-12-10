import itertools 
import os
import pickle

total_articles = []
total_combinations = []

directory = r'../public_processed_test_files'
for filename in os.listdir(directory):
    total_articles.append(int(filename[:-4]))

# print(total_articles)
amount = len(total_articles)
for i in range(amount):
    article_no = []
    article_no.append(total_articles.pop(0))
    combination = list(itertools.product(article_no, total_articles))
    total_combinations += combination

with open("../dict/first_stage_public_test.txt", "wb") as fp:
    pickle.dump(total_combinations, fp)