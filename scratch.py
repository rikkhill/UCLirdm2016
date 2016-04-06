# Scratchpad script

import numpy as np
import pandas as pd
from pymf import *
from collections import defaultdict

# Lets build a ratings matrix

K = 10

df = pd.read_csv("./data/small/ratings.csv")
# df = pd.read_csv("./data/1M/ratings.dat", sep='::')
df.columns = ['userId', 'movieId', 'rating', 'timestamp']


ratings = df.pivot(index="movieId", columns="userId", values="rating")

print(df["movieId"].unique().shape)

print(ratings.shape)

ratings.fillna(0, inplace=True)

rMatrix = ratings.as_matrix()

(d_m, d_n) = rMatrix.shape

# Lets make a bi-cross-validation weight matrix
hold_out_proportion = 0.5
m_indices = np.random.choice(d_m, int(d_m * hold_out_proportion), replace=False).tolist()
n_indices = np.random.choice(d_n, int(d_n * hold_out_proportion), replace=False).tolist()

weight_matrix = np.ones(rMatrix.shape)
weight_matrix[np.array(m_indices)[:, None], n_indices] = 0


def callout(arg):
    print(arg.frobenius_norm(complement=True))


nmf_model = WNMF(rMatrix, weight_matrix, num_bases=K)
nmf_model.factorize(niter=1, show_progress=True, epoch_hook=lambda x: callout(x))

movies = nmf_model.W

print(movies.shape)

base_movies = df["movieId"].unique().tolist()


# Get the tag relevance matrix

gr = pd.read_csv("./data/genome/tag_relevance.dat", header=None, sep='\t')
gr.columns = ["movieId", "tagId", "relevance"]

# Trim all movies that aren't in the base movies
gr = gr[gr["movieId"].isin(base_movies)]


gr_movies = set(gr["movieId"].unique().tolist())
setdiff = [bm for bm in base_movies if bm not in gr_movies]
empty_data = pd.DataFrame([(m_id, 0, 0) for m_id in setdiff])
empty_data.columns = ["movieId", "tagId", "relevance"]

gr = gr.append(empty_data)

print(gr)

# Pad out the pivot

"""
genome_relevance = genome_relevance.set_value(len(genome_relevance), max_movie_id, 1, 0)

relevance = genome_relevance.pivot(index="movieId", columns="tagId", values="relevance")
print(relevance.shape)

basis_examples = []

for i in range(0, K):
    col_array = np.asarray(movies[:, i])
    topten = col_array.argsort()[-10:][::-1]
    basis_examples.append(topten)

# Get movie data
movie_data = pd.read_csv("./data/small/movies.csv")

count = 1
for i in basis_examples:
    print("\nBasis %d" % count)
    count += 1
    for j in i:
        print(movie_data['title'].iloc[j])
"""