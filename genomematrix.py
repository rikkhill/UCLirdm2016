# Adds a csv file to the data directory of a specified dataset
# with relevance tags using that data directory's movie IDs

import pandas as pd
import numpy as np
from collections import defaultdict


base_dir = "./data/1M/"


base_movies = pd.read_csv(base_dir + "movies.dat", sep="::", header=None, engine='python')
base_movies.columns = ["movieId", "title", "genre"]

rel_matrix = np.loadtxt("./data/outputk10/dimrel.csv", delimiter=",")
movie_matrix = np.loadtxt("./data/outputk10/dimmovie.csv", delimiter=",")

tags = pd.read_csv("./data/genome/tags.dat", sep="\t", header=None)
tags.columns = ["tagId", "tag", "popularity"]

tr = pd.read_csv("./data/genome/tag_relevance.dat", sep="\t", header=None)
tr.columns = ["movieId", "tagId", "relevance"]

# We now need to remove all the movies from base_movies which
# aren't in the ratings matrix, because otherwise the indices
# won't align and it'll tell us Apocalypse Now is a cute
# animated children's film...
ratings = pd.read_csv(base_dir + "ratings.dat", sep="::", header=None, engine='python')
ratings.columns = ['userId', 'movieId', 'rating', 'timestamp']
rated_movies = ratings['movieId'].unique().tolist()
base_movies = base_movies[base_movies['movieId'].isin(rated_movies)]

(K, N) = rel_matrix.shape
print(K, N)
basis_tags = []
movie_examples = []

# Get examples from movies and examples from tags for each basis
for i in range(0, K):
    row_array = np.asarray(rel_matrix[i, :])
    toptwenty_tags = row_array.argsort()[-30:][::-1]
    basis_tags.append(toptwenty_tags)

    col_array = np.asarray(movie_matrix[:, i])
    topten_movies = col_array.argsort()[-10:][::-1]
    movie_examples.append(topten_movies)

movie_titles = []
for m in movie_examples:
    movie_titles.append([base_movies['title'].iloc[n] for n in m])

# Get a list of lists of tags for each basis

tag_counts = defaultdict(int)
tag_words = [[] for i in range(0, K)]
count = 1
for i in basis_tags:
    count += 1
    for j in i:
        word = tags['tag'].iloc[j]
        tag_words[count - 2].append(word)
        tag_counts[word] += 1

best_tag_words = []
for t in tag_words:
    scoredict = dict((k, tag_counts[k]) for k in t if k in tag_counts)
    # find 5 lowest values in dictionary
    best_words = []
    for j in range(0, 10):
        w = min(scoredict, key=scoredict.get)
        best_words.append(w)
        del scoredict[w]
    best_tag_words.append(best_words)

for i in range(0, K):
    print("\nBasis %d" % (i + 1))
    print(best_tag_words[i])
    print(movie_titles[i])

