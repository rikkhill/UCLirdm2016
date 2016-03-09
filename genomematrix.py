# Adds a csv file to the data directory of a specified dataset
# with relevance tags using that data directory's movie IDs

import pandas as pd
import numpy as np


base_dir = "./data/small/"

# load in genome movie list

base_movies = pd.read_csv(base_dir + "movies.csv")
reverse_map = dict(list(zip(base_movies.title, base_movies.movieId)))
print(reverse_map)

"""
genome_movies = pd.read_csv("./data/genome/movies.dat", header=None, sep='\t')
genome_movies.columns = ["id", "title", "popularity"]
reverse_map = dict(list(zip(genome_movies.title, genome_movies.id)))
"""




